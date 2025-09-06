/**
 * RAPTOR Offline Builder
 * 
 * Builds hierarchical RAPTOR trees from semantic cards using mini-batch k-means
 * clustering and generates summary embeddings for each node.
 */

import { SemanticCard } from './semantic-card.js';
import { RaptorSnapshot, RaptorNode, RaptorSnapshotConfig, RaptorSnapshotManager } from './snapshot.js';
import { RaptorEmbeddingService, BusinessnessScorer } from './embeddings.js';

export interface ClusteringConfig {
  maxIterations: number;
  convergenceThreshold: number;
  miniBatchSize: number;
  randomSeed?: number;
}

export interface BuildProgress {
  stage: 'extracting' | 'clustering' | 'summarizing' | 'persisting' | 'complete';
  progress: number; // 0-100
  details: string;
}

export interface BuildResult {
  snapshot: RaptorSnapshot;
  metrics: BuildMetrics;
  warnings: string[];
}

export interface BuildMetrics {
  totalFiles: number;
  totalNodes: number;
  levels: number;
  clusteringTime: number;
  summarizationTime: number;
  embeddingTime: number;
  avgBusinessnessScore: number;
  clusterBalance: number[]; // Files per cluster at each level
}

/**
 * Mini-batch K-means clustering implementation
 */
class MiniBatchKMeans {
  private k: number;
  private maxIterations: number;
  private convergenceThreshold: number;
  private batchSize: number;
  private centroids: number[][];
  private assignments: number[];
  private random: () => number;

  constructor(
    k: number,
    config: ClusteringConfig = {
      maxIterations: 100,
      convergenceThreshold: 0.001,
      miniBatchSize: 100
    }
  ) {
    this.k = k;
    this.maxIterations = config.maxIterations;
    this.convergenceThreshold = config.convergenceThreshold;
    this.batchSize = config.miniBatchSize;
    this.centroids = [];
    this.assignments = [];
    
    // Seeded random for reproducibility
    const seed = config.randomSeed || 42;
    let currentSeed = seed;
    this.random = () => {
      currentSeed = (currentSeed * 9301 + 49297) % 233280;
      return currentSeed / 233280;
    };
  }

  fit(embeddings: number[][]): number[] {
    if (embeddings.length === 0) return [];
    
    const dimension = embeddings[0].length;
    this.initializeCentroids(embeddings, dimension);
    this.assignments = new Array(embeddings.length).fill(0);

    let converged = false;
    let iteration = 0;

    while (!converged && iteration < this.maxIterations) {
      const oldCentroids = this.centroids.map(centroid => [...centroid]);
      
      // Process mini-batches
      const shuffledIndices = this.shuffle([...Array(embeddings.length).keys()]);
      
      for (let i = 0; i < shuffledIndices.length; i += this.batchSize) {
        const batchIndices = shuffledIndices.slice(i, i + this.batchSize);
        const batch = batchIndices.map(idx => embeddings[idx]);
        
        this.updateCentroidsWithBatch(batch, batchIndices);
      }

      // Check convergence
      converged = this.hasConverged(oldCentroids, this.centroids);
      iteration++;
    }

    // Final assignment pass
    this.assignments = embeddings.map(embedding => this.findNearestCentroid(embedding));
    
    return this.assignments;
  }

  private initializeCentroids(embeddings: number[][], dimension: number): void {
    // K-means++ initialization
    this.centroids = [];
    
    if (this.k >= embeddings.length) {
      // If k >= number of points, use all points as centroids
      this.centroids = embeddings.slice(0, this.k).map(e => [...e]);
      return;
    }

    // Choose first centroid randomly
    const firstIndex = Math.floor(this.random() * embeddings.length);
    this.centroids.push([...embeddings[firstIndex]]);

    // Choose remaining centroids using k-means++
    for (let i = 1; i < this.k; i++) {
      const distances = embeddings.map(embedding => {
        const minDistToCentroid = Math.min(
          ...this.centroids.map(centroid => this.euclideanDistance(embedding, centroid))
        );
        return minDistToCentroid * minDistToCentroid;
      });

      const totalDistance = distances.reduce((sum, d) => sum + d, 0);
      const threshold = this.random() * totalDistance;

      let cumulative = 0;
      for (let j = 0; j < distances.length; j++) {
        cumulative += distances[j];
        if (cumulative >= threshold) {
          this.centroids.push([...embeddings[j]]);
          break;
        }
      }
    }
  }

  private updateCentroidsWithBatch(batch: number[][], batchIndices: number[]): void {
    const clusterCounts = new Array(this.k).fill(0);
    const clusterSums = this.centroids.map(centroid => new Array(centroid.length).fill(0));

    // Assign points in batch to nearest centroids
    batch.forEach((embedding, i) => {
      const clusterIndex = this.findNearestCentroid(embedding);
      clusterCounts[clusterIndex]++;
      
      for (let d = 0; d < embedding.length; d++) {
        clusterSums[clusterIndex][d] += embedding[d];
      }
    });

    // Update centroids with learning rate decay
    const learningRate = 0.1; // Could be adaptive
    
    for (let c = 0; c < this.k; c++) {
      if (clusterCounts[c] > 0) {
        const newCentroid = clusterSums[c].map(sum => sum / clusterCounts[c]);
        
        // Weighted update: old_centroid * (1 - lr) + new_centroid * lr
        for (let d = 0; d < this.centroids[c].length; d++) {
          this.centroids[c][d] = this.centroids[c][d] * (1 - learningRate) + 
                                newCentroid[d] * learningRate;
        }
      }
    }
  }

  private findNearestCentroid(embedding: number[]): number {
    let minDistance = Infinity;
    let nearestIndex = 0;

    for (let i = 0; i < this.centroids.length; i++) {
      const distance = this.euclideanDistance(embedding, this.centroids[i]);
      if (distance < minDistance) {
        minDistance = distance;
        nearestIndex = i;
      }
    }

    return nearestIndex;
  }

  private euclideanDistance(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  private hasConverged(oldCentroids: number[][], newCentroids: number[][]): boolean {
    for (let i = 0; i < oldCentroids.length; i++) {
      const distance = this.euclideanDistance(oldCentroids[i], newCentroids[i]);
      if (distance > this.convergenceThreshold) {
        return false;
      }
    }
    return true;
  }

  private shuffle<T>(array: T[]): T[] {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(this.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }

  getCentroids(): number[][] {
    return this.centroids.map(centroid => [...centroid]);
  }
}

/**
 * RAPTOR tree builder
 */
export class RaptorBuilder {
  private embeddingService: RaptorEmbeddingService;
  private businessnessScorer: BusinessnessScorer;
  private snapshotManager: RaptorSnapshotManager;

  constructor(
    embeddingService: RaptorEmbeddingService,
    businessnessScorer: BusinessnessScorer,
    snapshotManager: RaptorSnapshotManager
  ) {
    this.embeddingService = embeddingService;
    this.businessnessScorer = businessnessScorer;
    this.snapshotManager = snapshotManager;
  }

  async buildSnapshot(
    repoSha: string,
    cards: SemanticCard[],
    config?: RaptorSnapshotConfig,
    progressCallback?: (progress: BuildProgress) => void
  ): Promise<BuildResult> {
    const startTime = Date.now();
    const warnings: string[] = [];
    
    if (cards.length === 0) {
      throw new Error('Cannot build RAPTOR snapshot with no semantic cards');
    }

    // Use default config if not provided
    const snapshotConfig = config || RaptorSnapshotManager.createDefaultConfig();
    
    // Compute clustering parameters
    const { k1, k2 } = RaptorSnapshotManager.computeClusterParams(cards.length);
    snapshotConfig.k1 = k1;
    snapshotConfig.k2 = k2;

    progressCallback?.({ stage: 'extracting', progress: 10, details: 'Computing embeddings for semantic cards' });

    // Compute embeddings for all cards
    const embeddingTime = Date.now();
    const cardEmbeddings = new Map<string, number[]>();
    
    for (let i = 0; i < cards.length; i++) {
      const card = cards[i];
      const embedding = await this.embeddingService.embedSemanticCard(card);
      cardEmbeddings.set(card.file_id, embedding);
      
      if (i % 50 === 0) {
        progressCallback?.({ 
          stage: 'extracting', 
          progress: 10 + (i / cards.length) * 20, 
          details: `Embedded ${i}/${cards.length} cards` 
        });
      }
    }

    const embeddingElapsed = Date.now() - embeddingTime;

    progressCallback?.({ stage: 'clustering', progress: 30, details: 'Building hierarchical clusters' });

    // Build hierarchical clustering
    const clusteringTime = Date.now();
    const snapshot = this.snapshotManager.createEmptySnapshot(repoSha, snapshotConfig);
    
    await this.buildHierarchicalClusters(
      cards,
      cardEmbeddings,
      snapshot,
      progressCallback
    );
    
    const clusteringElapsed = Date.now() - clusteringTime;

    progressCallback?.({ stage: 'summarizing', progress: 70, details: 'Generating node summaries' });

    // Generate summaries and summary embeddings
    const summarizationTime = Date.now();
    await this.generateNodeSummaries(snapshot, cards, progressCallback);
    const summarizationElapsed = Date.now() - summarizationTime;

    progressCallback?.({ stage: 'persisting', progress: 90, details: 'Saving snapshot' });

    // Validate snapshot
    const validation = this.snapshotManager.validateSnapshot(snapshot);
    if (!validation.isValid) {
      throw new Error(`Invalid snapshot: ${validation.errors.join(', ')}`);
    }

    // Compute metrics
    const metrics = this.computeBuildMetrics(
      snapshot,
      cards,
      clusteringElapsed,
      summarizationElapsed,
      embeddingElapsed
    );

    progressCallback?.({ stage: 'complete', progress: 100, details: 'Build complete' });

    return {
      snapshot,
      metrics,
      warnings
    };
  }

  private async buildHierarchicalClusters(
    cards: SemanticCard[],
    cardEmbeddings: Map<string, number[]>,
    snapshot: RaptorSnapshot,
    progressCallback?: (progress: BuildProgress) => void
  ): Promise<void> {
    const embeddings = cards.map(card => cardEmbeddings.get(card.file_id)!);
    let currentLevelNodes = cards.map((card, i) => {
      const nodeId = `leaf-${i}`;
      const node: RaptorNode = {
        id: nodeId,
        level: 0,
        children: [],
        centroid: embeddings[i],
        summary: '', // Will be filled in later
        summaryEmb: [],
        file_ids: [card.file_id],
        last_update_ts: Date.now(),
        pressure: 0
      };
      
      snapshot.nodes.set(nodeId, node);
      snapshot.fileToLeaf.set(card.file_id, [nodeId]);
      return node;
    });

    // Build levels bottom-up
    let level = 1;
    while (currentLevelNodes.length > 1 && level <= snapshot.config.max_levels) {
      progressCallback?.({ 
        stage: 'clustering', 
        progress: 30 + (level / snapshot.config.max_levels) * 30, 
        details: `Building level ${level} with ${currentLevelNodes.length} nodes` 
      });

      const k = level === 1 ? snapshot.config.k1 : snapshot.config.k2;
      const clusterCount = Math.min(k, Math.floor(currentLevelNodes.length / 2));
      
      if (clusterCount <= 1) break;

      const nodeEmbeddings = currentLevelNodes.map(node => node.centroid);
      const kmeans = new MiniBatchKMeans(clusterCount);
      const assignments = kmeans.fit(nodeEmbeddings);

      // Create parent nodes
      const parentNodes: RaptorNode[] = [];
      
      for (let clusterId = 0; clusterId < clusterCount; clusterId++) {
        const childNodes = currentLevelNodes.filter((_, i) => assignments[i] === clusterId);
        
        if (childNodes.length === 0) continue;

        const parentId = `level-${level}-cluster-${clusterId}`;
        const childEmbeddings = childNodes.map(child => child.centroid);
        const centroid = this.embeddingService.computeCentroid(childEmbeddings);
        
        // Collect all files in subtree
        const fileIds = new Set<string>();
        for (const child of childNodes) {
          for (const fileId of child.file_ids) {
            fileIds.add(fileId);
          }
        }

        const parentNode: RaptorNode = {
          id: parentId,
          level,
          children: childNodes.map(child => child.id),
          centroid,
          summary: '', // Will be filled in later
          summaryEmb: [],
          file_ids: Array.from(fileIds),
          last_update_ts: Date.now(),
          pressure: 0
        };

        // Set parent relationships
        for (const child of childNodes) {
          child.parent = parentId;
        }

        snapshot.nodes.set(parentId, parentNode);
        parentNodes.push(parentNode);
      }

      currentLevelNodes = parentNodes;
      level++;
    }
  }

  private async generateNodeSummaries(
    snapshot: RaptorSnapshot,
    cards: SemanticCard[],
    progressCallback?: (progress: BuildProgress) => void
  ): Promise<void> {
    const cardMap = new Map(cards.map(card => [card.file_id, card]));
    const nodes = Array.from(snapshot.nodes.values());
    
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      
      if (i % 20 === 0) {
        progressCallback?.({ 
          stage: 'summarizing', 
          progress: 70 + (i / nodes.length) * 20, 
          details: `Summarizing node ${i}/${nodes.length}` 
        });
      }

      // Generate summary from cards in this node's subtree
      const nodeCards = node.file_ids
        .map(fileId => cardMap.get(fileId))
        .filter(card => card !== undefined) as SemanticCard[];

      node.summary = this.generateNodeSummary(nodeCards, snapshot.config.summary_max_tokens);
      
      if (node.summary) {
        node.summaryEmb = await this.embeddingService.embedSummary(node.summary);
      } else {
        node.summaryEmb = new Array(this.embeddingService.computeCentroid([]).length).fill(0);
      }
    }
  }

  private generateNodeSummary(cards: SemanticCard[], maxTokens: number): string {
    if (cards.length === 0) return '';

    // Aggregate features across all cards in the node
    const allRoles = new Set<string>();
    const allRoutes = new Set<string>();
    const allTables = new Set<string>();
    const allTopics = new Set<string>();
    const allTypes = new Set<string>();
    const allDomainTokens = new Set<string>();
    const allEffects = new Set<string>();

    for (const card of cards) {
      card.roles.forEach(role => allRoles.add(role));
      card.resources.routes.forEach(route => allRoutes.add(route));
      card.resources.sql.forEach(table => allTables.add(table));
      card.resources.topics.forEach(topic => allTopics.add(topic));
      card.shapes.typeNames.forEach(type => allTypes.add(type));
      card.domainTokens.slice(0, 5).forEach(token => allDomainTokens.add(token));
      card.effects.forEach(effect => allEffects.add(effect));
    }

    // Build bullet list summary
    const summaryParts: string[] = [];

    if (allRoles.size > 0) {
      summaryParts.push(`• Roles: ${Array.from(allRoles).slice(0, 3).join(', ')}`);
    }

    if (allRoutes.size > 0) {
      summaryParts.push(`• Routes: ${Array.from(allRoutes).slice(0, 3).join(', ')}`);
    }

    if (allTables.size > 0) {
      summaryParts.push(`• Tables: ${Array.from(allTables).slice(0, 3).join(', ')}`);
    }

    if (allTopics.size > 0) {
      summaryParts.push(`• Topics: ${Array.from(allTopics).slice(0, 3).join(', ')}`);
    }

    if (allTypes.size > 0) {
      summaryParts.push(`• Types: ${Array.from(allTypes).slice(0, 5).join(', ')}`);
    }

    if (allDomainTokens.size > 0) {
      summaryParts.push(`• Domain: ${Array.from(allDomainTokens).slice(0, 8).join(', ')}`);
    }

    if (allEffects.size > 0) {
      summaryParts.push(`• Effects: ${Array.from(allEffects).join(', ')}`);
    }

    // Add file count
    summaryParts.push(`• Files: ${cards.length}`);

    let summary = summaryParts.join('\n');

    // Truncate if too long (rough token estimation: ~4 chars per token)
    if (summary.length > maxTokens * 4) {
      summary = summary.substring(0, maxTokens * 4 - 3) + '...';
    }

    return summary;
  }

  private computeBuildMetrics(
    snapshot: RaptorSnapshot,
    cards: SemanticCard[],
    clusteringTime: number,
    summarizationTime: number,
    embeddingTime: number
  ): BuildMetrics {
    const levels = Math.max(...Array.from(snapshot.nodes.values()).map(node => node.level)) + 1;
    const avgBusinessness = cards.reduce((sum, card) => sum + card.B, 0) / cards.length;

    // Compute cluster balance per level
    const clusterBalance: number[] = [];
    for (let level = 0; level < levels; level++) {
      const levelNodes = Array.from(snapshot.nodes.values()).filter(node => node.level === level);
      const filesPerNode = levelNodes.map(node => node.file_ids.length);
      const avgFilesPerNode = filesPerNode.reduce((sum, count) => sum + count, 0) / filesPerNode.length;
      clusterBalance.push(avgFilesPerNode || 0);
    }

    return {
      totalFiles: cards.length,
      totalNodes: snapshot.nodes.size,
      levels,
      clusteringTime,
      summarizationTime,
      embeddingTime,
      avgBusinessnessScore: avgBusinessness,
      clusterBalance
    };
  }
}

export default RaptorBuilder;