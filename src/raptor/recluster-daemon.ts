/**
 * Pressure-Driven Partial Reclustering Daemon
 * 
 * Maintains RAPTOR tree freshness through selective reclustering based on
 * pressure metrics: edit churn, embedding drift, query heat, and age.
 */

import { RaptorSnapshot, RaptorNode, PressureMetrics, PressureWeights, ReclusterBudget, RaptorSnapshotManager } from './snapshot.js';
import { SemanticCard } from './semantic-card.js';
import { RaptorEmbeddingService } from './embeddings.js';
import { RaptorBuilder } from './builder.js';
import EventEmitter from 'events';

export interface DaemonConfig {
  enabled: boolean;
  intervalMinutes: number;     // How often to run (default: 60)
  pressureWeights: PressureWeights;
  budget: ReclusterBudget;
  ttlDays: number;            // Nodes older than this are considered stale
  hysteresis: number;         // Minimum centroid drift to trigger ancestor updates (0.05)
  backlogThreshold: number;   // Warn if backlog > this * hourly budget (3.0)
}

export interface DaemonStatus {
  enabled: boolean;
  lastRun: number;
  nextRun: number;
  currentBudget: ReclusterBudget;
  pressureStats: PressureStats;
  backlogSize: number;
  warnings: string[];
}

export interface PressureStats {
  totalNodes: number;
  highPressureNodes: number;  // > 75th percentile
  avgPressure: number;
  maxPressure: number;
  pressureDistribution: number[]; // Histogram buckets
}

export interface ReclusterOperation {
  nodeId: string;
  pressure: number;
  estimatedCost: OperationCost;
  type: 'recluster' | 'summarize' | 'embed';
}

export interface OperationCost {
  summaries: number;
  cpuSeconds: number;
  embeddings: number;
}

export interface ReclusterResult {
  nodeId: string;
  success: boolean;
  costUsed: OperationCost;
  ancestorsUpdated: string[];
  error?: string;
}

/**
 * EWMA (Exponentially Weighted Moving Average) tracker
 */
class EWMATracker {
  private value: number;
  private alpha: number; // Decay factor

  constructor(initialValue: number = 0, alpha: number = 0.1) {
    this.value = initialValue;
    this.alpha = alpha;
  }

  update(newValue: number): void {
    this.value = this.alpha * newValue + (1 - this.alpha) * this.value;
  }

  getValue(): number {
    return this.value;
  }

  reset(): void {
    this.value = 0;
  }
}

/**
 * Tracks pressure metrics for RAPTOR nodes
 */
class PressureTracker {
  private editChurnTrackers: Map<string, EWMATracker>;
  private lastEmbeddings: Map<string, number[]>;
  private queryHeatTrackers: Map<string, EWMATracker>;
  private nodeCreationTimes: Map<string, number>;

  constructor() {
    this.editChurnTrackers = new Map();
    this.lastEmbeddings = new Map();
    this.queryHeatTrackers = new Map();
    this.nodeCreationTimes = new Map();
  }

  recordEdit(nodeId: string, bytesChanged: number): void {
    if (!this.editChurnTrackers.has(nodeId)) {
      this.editChurnTrackers.set(nodeId, new EWMATracker());
    }
    this.editChurnTrackers.get(nodeId)!.update(bytesChanged);
  }

  recordQueryHit(nodeId: string): void {
    if (!this.queryHeatTrackers.has(nodeId)) {
      this.queryHeatTrackers.set(nodeId, new EWMATracker());
    }
    this.queryHeatTrackers.get(nodeId)!.update(1);
  }

  updateNodeEmbedding(nodeId: string, embedding: number[]): void {
    this.lastEmbeddings.set(nodeId, [...embedding]);
  }

  setNodeCreationTime(nodeId: string, timestamp: number): void {
    this.nodeCreationTimes.set(nodeId, timestamp);
  }

  computePressureMetrics(nodeId: string, currentEmbedding: number[]): PressureMetrics {
    const editChurn = this.editChurnTrackers.get(nodeId)?.getValue() || 0;
    const queryHeat = this.queryHeatTrackers.get(nodeId)?.getValue() || 0;
    
    let embeddingDrift = 0;
    if (this.lastEmbeddings.has(nodeId)) {
      const lastEmbedding = this.lastEmbeddings.get(nodeId)!;
      embeddingDrift = this.computeL2Distance(currentEmbedding, lastEmbedding);
    }

    const creationTime = this.nodeCreationTimes.get(nodeId) || Date.now();
    const age = (Date.now() - creationTime) / (1000 * 60 * 60); // Hours

    return { editChurn, embeddingDrift, queryHeat, age };
  }

  private computeL2Distance(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;
    
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  decayHeat(): void {
    // Periodic decay of query heat
    for (const tracker of this.queryHeatTrackers.values()) {
      tracker.update(0);
    }
  }

  cleanup(activeNodeIds: Set<string>): void {
    // Remove trackers for deleted nodes
    const allKeys = [
      ...this.editChurnTrackers.keys(),
      ...this.lastEmbeddings.keys(),
      ...this.queryHeatTrackers.keys(),
      ...this.nodeCreationTimes.keys()
    ];

    for (const nodeId of allKeys) {
      if (!activeNodeIds.has(nodeId)) {
        this.editChurnTrackers.delete(nodeId);
        this.lastEmbeddings.delete(nodeId);
        this.queryHeatTrackers.delete(nodeId);
        this.nodeCreationTimes.delete(nodeId);
      }
    }
  }
}

/**
 * Main reclustering daemon
 */
export class ReclusterDaemon extends EventEmitter {
  private config: DaemonConfig;
  private snapshotManager: RaptorSnapshotManager;
  private embeddingService: RaptorEmbeddingService;
  private builder: RaptorBuilder;
  private pressureTracker: PressureTracker;
  private isRunning: boolean;
  private intervalId?: NodeJS.Timeout;
  private currentSnapshots: Map<string, RaptorSnapshot>;

  constructor(
    config: DaemonConfig,
    snapshotManager: RaptorSnapshotManager,
    embeddingService: RaptorEmbeddingService,
    builder: RaptorBuilder
  ) {
    super();
    this.config = config;
    this.snapshotManager = snapshotManager;
    this.embeddingService = embeddingService;
    this.builder = builder;
    this.pressureTracker = new PressureTracker();
    this.isRunning = false;
    this.currentSnapshots = new Map();
  }

  static createDefaultConfig(): DaemonConfig {
    return {
      enabled: true,
      intervalMinutes: 60,
      pressureWeights: { wc: 0.4, wd: 0.3, wq: 0.2, wa: 0.1 },
      budget: {
        max_summaries_per_hour: 200,
        max_cpu_seconds_per_hour: 300,
        current_summaries_used: 0,
        current_cpu_used: 0,
        reset_ts: Date.now()
      },
      ttlDays: 14,
      hysteresis: 0.05,
      backlogThreshold: 3.0
    };
  }

  start(): void {
    if (this.isRunning) return;
    
    this.isRunning = true;
    this.emit('daemon-started', { config: this.config });

    if (this.config.enabled) {
      this.scheduleNextRun();
    }
  }

  stop(): void {
    if (!this.isRunning) return;
    
    this.isRunning = false;
    if (this.intervalId) {
      clearTimeout(this.intervalId);
      this.intervalId = undefined;
    }
    
    this.emit('daemon-stopped');
  }

  updateConfig(newConfig: Partial<DaemonConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.emit('config-updated', { config: this.config });

    // Restart with new config if enabled changed
    if (this.isRunning) {
      this.stop();
      this.start();
    }
  }

  // External event recording
  recordFileEdit(repoSha: string, filePath: string, bytesChanged: number): void {
    // Map file to nodes and record edit churn
    const snapshot = this.currentSnapshots.get(repoSha);
    if (!snapshot) return;

    const affectedNodes = this.findNodesContainingFile(snapshot, filePath);
    for (const nodeId of affectedNodes) {
      this.pressureTracker.recordEdit(nodeId, bytesChanged);
    }
  }

  recordQueryHit(repoSha: string, nodeId: string): void {
    this.pressureTracker.recordQueryHit(nodeId);
  }

  loadSnapshot(snapshot: RaptorSnapshot): void {
    this.currentSnapshots.set(snapshot.repo_sha, snapshot);
    
    // Initialize tracking for all nodes
    for (const [nodeId, node] of snapshot.nodes) {
      this.pressureTracker.setNodeCreationTime(nodeId, node.last_update_ts);
      this.pressureTracker.updateNodeEmbedding(nodeId, node.centroid);
    }
  }

  getStatus(): DaemonStatus {
    const now = Date.now();
    let totalPressure = 0;
    let pressureValues: number[] = [];

    for (const snapshot of this.currentSnapshots.values()) {
      for (const [nodeId, node] of snapshot.nodes) {
        const metrics = this.pressureTracker.computePressureMetrics(nodeId, node.centroid);
        const pressure = RaptorSnapshotManager.calculatePressure(node, metrics, this.config.pressureWeights);
        totalPressure += pressure;
        pressureValues.push(pressure);
      }
    }

    pressureValues.sort((a, b) => a - b);
    const p75Index = Math.floor(pressureValues.length * 0.75);
    const p75Threshold = pressureValues[p75Index] || 0;
    const highPressureNodes = pressureValues.filter(p => p > p75Threshold).length;

    // Check budget reset
    if (now - this.config.budget.reset_ts > 60 * 60 * 1000) {
      this.config.budget.current_summaries_used = 0;
      this.config.budget.current_cpu_used = 0;
      this.config.budget.reset_ts = now;
    }

    const backlogSize = this.estimateBacklogSize();
    const warnings = this.checkWarnings(backlogSize);

    return {
      enabled: this.config.enabled,
      lastRun: 0, // TODO: Track last run time
      nextRun: 0, // TODO: Calculate next run time
      currentBudget: { ...this.config.budget },
      pressureStats: {
        totalNodes: pressureValues.length,
        highPressureNodes,
        avgPressure: totalPressure / Math.max(pressureValues.length, 1),
        maxPressure: Math.max(...pressureValues, 0),
        pressureDistribution: this.computeHistogram(pressureValues)
      },
      backlogSize,
      warnings
    };
  }

  private scheduleNextRun(): void {
    if (!this.config.enabled) return;

    const intervalMs = this.config.intervalMinutes * 60 * 1000;
    this.intervalId = setTimeout(() => {
      this.runReclusterCycle();
      this.scheduleNextRun();
    }, intervalMs);
  }

  private async runReclusterCycle(): Promise<void> {
    if (!this.config.enabled) return;

    this.emit('cycle-started');

    try {
      // Decay query heat
      this.pressureTracker.decayHeat();

      // Process each snapshot
      for (const [repoSha, snapshot] of this.currentSnapshots) {
        await this.processSnapshot(repoSha, snapshot);
      }

      // Clean up tracking for deleted nodes
      const allActiveNodes = new Set<string>();
      for (const snapshot of this.currentSnapshots.values()) {
        for (const nodeId of snapshot.nodes.keys()) {
          allActiveNodes.add(nodeId);
        }
      }
      this.pressureTracker.cleanup(allActiveNodes);

      this.emit('cycle-completed');
    } catch (error) {
      this.emit('cycle-error', error);
    }
  }

  private async processSnapshot(repoSha: string, snapshot: RaptorSnapshot): Promise<void> {
    // Compute pressure for all nodes
    const operations: ReclusterOperation[] = [];
    
    for (const [nodeId, node] of snapshot.nodes) {
      const metrics = this.pressureTracker.computePressureMetrics(nodeId, node.centroid);
      const pressure = RaptorSnapshotManager.calculatePressure(node, metrics, this.config.pressureWeights);
      
      // Update node pressure
      node.pressure = pressure;

      // Estimate cost of reclustering this node
      const cost = this.estimateOperationCost(node, snapshot);
      
      operations.push({
        nodeId,
        pressure,
        estimatedCost: cost,
        type: 'recluster'
      });
    }

    // Sort by pressure (descending) and process while budget allows
    operations.sort((a, b) => b.pressure - a.pressure);

    for (const operation of operations) {
      if (!this.hasBudgetFor(operation.estimatedCost)) {
        break;
      }

      const result = await this.executeReclusterOperation(repoSha, snapshot, operation);
      this.emit('operation-completed', { repoSha, operation, result });

      if (result.success) {
        this.consumeBudget(result.costUsed);
        this.resetPressure(operation.nodeId, snapshot);
      }
    }
  }

  private async executeReclusterOperation(
    repoSha: string,
    snapshot: RaptorSnapshot,
    operation: ReclusterOperation
  ): Promise<ReclusterResult> {
    const startTime = Date.now();
    const startCpu = process.cpuUsage();

    try {
      const node = snapshot.nodes.get(operation.nodeId)!;
      
      // Re-embed leaves in this subtree
      await this.reembedLeaves(node, snapshot);
      
      // Perform mini-batch k-means on immediate children
      await this.reclusterNode(node, snapshot);
      
      // Update ancestors until drift < hysteresis
      const ancestorsUpdated = await this.updateAncestors(node, snapshot);

      const endCpu = process.cpuUsage(startCpu);
      const cpuSeconds = (endCpu.user + endCpu.system) / 1000000; // Convert to seconds

      return {
        nodeId: operation.nodeId,
        success: true,
        costUsed: {
          summaries: 1, // Simplified
          cpuSeconds,
          embeddings: node.children.length
        },
        ancestorsUpdated
      };
    } catch (error) {
      return {
        nodeId: operation.nodeId,
        success: false,
        costUsed: { summaries: 0, cpuSeconds: 0, embeddings: 0 },
        ancestorsUpdated: [],
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  private async reembedLeaves(node: RaptorNode, snapshot: RaptorSnapshot): Promise<void> {
    // For leaf nodes, re-embed based on current file content
    if (node.children.length === 0) {
      // In a real implementation, this would reload the file and re-extract features
      // For now, we'll just add some noise to simulate embedding drift
      node.centroid = node.centroid.map(x => x + (Math.random() - 0.5) * 0.01);
      this.pressureTracker.updateNodeEmbedding(node.id, node.centroid);
    }
  }

  private async reclusterNode(node: RaptorNode, snapshot: RaptorSnapshot): Promise<void> {
    if (node.children.length <= 1) return;

    // Get child centroids
    const childEmbeddings = node.children
      .map(childId => snapshot.nodes.get(childId))
      .filter(child => child !== undefined)
      .map(child => child!.centroid);

    if (childEmbeddings.length > 0) {
      // Recompute centroid
      node.centroid = this.embeddingService.computeCentroid(childEmbeddings);
      node.last_update_ts = Date.now();
    }
  }

  private async updateAncestors(node: RaptorNode, snapshot: RaptorSnapshot): Promise<string[]> {
    const updated: string[] = [];
    let current = node.parent ? snapshot.nodes.get(node.parent) : undefined;

    while (current) {
      const oldCentroid = [...current.centroid];
      
      // Recompute centroid from children
      const childEmbeddings = current.children
        .map(childId => snapshot.nodes.get(childId))
        .filter(child => child !== undefined)
        .map(child => child!.centroid);

      if (childEmbeddings.length > 0) {
        current.centroid = this.embeddingService.computeCentroid(childEmbeddings);
        current.last_update_ts = Date.now();

        // Check if drift exceeds hysteresis threshold
        const drift = this.embeddingService.cosineSimilarity(oldCentroid, current.centroid);
        if (1 - drift < this.config.hysteresis) {
          break; // Stop propagating updates
        }

        updated.push(current.id);
        current = current.parent ? snapshot.nodes.get(current.parent) : undefined;
      } else {
        break;
      }
    }

    return updated;
  }

  private estimateOperationCost(node: RaptorNode, snapshot: RaptorSnapshot): OperationCost {
    // Simple cost estimation
    const embeddings = Math.max(1, node.children.length);
    const summaries = 1;
    const cpuSeconds = embeddings * 0.1 + summaries * 0.5; // Rough estimates

    return { summaries, cpuSeconds, embeddings };
  }

  private hasBudgetFor(cost: OperationCost): boolean {
    return (
      this.config.budget.current_summaries_used + cost.summaries <= this.config.budget.max_summaries_per_hour &&
      this.config.budget.current_cpu_used + cost.cpuSeconds <= this.config.budget.max_cpu_seconds_per_hour
    );
  }

  private consumeBudget(cost: OperationCost): void {
    this.config.budget.current_summaries_used += cost.summaries;
    this.config.budget.current_cpu_used += cost.cpuSeconds;
  }

  private resetPressure(nodeId: string, snapshot: RaptorSnapshot): void {
    const node = snapshot.nodes.get(nodeId);
    if (node) {
      node.pressure = 0;
    }
  }

  private findNodesContainingFile(snapshot: RaptorSnapshot, filePath: string): string[] {
    // In practice, you'd use the fileId, not filePath
    const fileId = filePath; // Simplified
    return snapshot.fileToLeaf.get(fileId) || [];
  }

  private estimateBacklogSize(): number {
    let highPressureOps = 0;
    
    for (const snapshot of this.currentSnapshots.values()) {
      for (const node of snapshot.nodes.values()) {
        if (node.pressure > 1.0) { // Arbitrary threshold
          highPressureOps++;
        }
      }
    }

    return highPressureOps;
  }

  private checkWarnings(backlogSize: number): string[] {
    const warnings: string[] = [];

    // Check budget usage
    const summaryUsage = this.config.budget.current_summaries_used / this.config.budget.max_summaries_per_hour;
    const cpuUsage = this.config.budget.current_cpu_used / this.config.budget.max_cpu_seconds_per_hour;

    if (summaryUsage > 0.8) {
      warnings.push(`High summary budget usage: ${(summaryUsage * 100).toFixed(1)}%`);
    }

    if (cpuUsage > 0.8) {
      warnings.push(`High CPU budget usage: ${(cpuUsage * 100).toFixed(1)}%`);
    }

    // Check backlog
    const hourlyCapacity = Math.min(
      this.config.budget.max_summaries_per_hour,
      this.config.budget.max_cpu_seconds_per_hour / 0.5 // Assuming 0.5s per op
    );

    if (backlogSize > this.config.backlogThreshold * hourlyCapacity) {
      warnings.push(`High pressure backlog: ${backlogSize} operations pending`);
    }

    return warnings;
  }

  private computeHistogram(values: number[], bins: number = 10): number[] {
    if (values.length === 0) return new Array(bins).fill(0);

    const min = Math.min(...values);
    const max = Math.max(...values);
    const binWidth = (max - min) / bins;
    const histogram = new Array(bins).fill(0);

    for (const value of values) {
      const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
      histogram[binIndex]++;
    }

    return histogram;
  }
}

export default ReclusterDaemon;