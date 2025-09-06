/**
 * Runtime RAPTOR Features for Stage-C Pipeline
 * 
 * Computes RAPTOR-based features for candidate files during query processing:
 * - raptor_max_sim: Maximum similarity to ancestors
 * - raptor_depth_of_best: Depth of best matching ancestor
 * - topic_overlap: Jaccard similarity of query and card features
 * - B: Businessness score from semantic card
 */

import { RaptorSnapshot, RaptorNode } from './snapshot.js';
import { SemanticCard } from './semantic-card.js';
import { RaptorEmbeddingService } from './embeddings.js';

export interface RaptorFeatures {
  raptor_max_sim: number;      // [0, 1] - max cosine similarity to ancestors
  raptor_depth_of_best: number; // {1, 2, 3} - depth of best matching ancestor
  topic_overlap: number;       // [0, 1] - Jaccard similarity of features
  B: number;                   // businessness score from semantic card
}

export interface QueryEmbedding {
  semantic: number[];          // Query semantic embedding
  features: QueryFeatures;     // Extracted query features
}

export interface QueryFeatures {
  routes: Set<string>;         // Route patterns mentioned in query
  tables: Set<string>;         // Table names mentioned in query
  types: Set<string>;          // Type names mentioned in query
  tokens: Set<string>;         // Domain tokens in query
  effects: Set<string>;        // Side effects mentioned
}

export interface FeatureComputationConfig {
  enabled: boolean;
  nlThreshold: number;         // Minimum NL score to compute RAPTOR features
  maxCandidates: number;       // Max Stage-A candidates to trigger computation
  priorBoostCap: number;       // Maximum log-odds boost for path prior (0.5)
  topicThreshold: number;      // Minimum similarity for path prior boost (0.35)
}

export interface PathPriorBoost {
  fileId: string;
  boost: number;              // Log-odds boost to apply
  reason: string;             // Why this boost was applied
}

/**
 * Query feature extractor
 */
export class QueryFeatureExtractor {
  private routePatterns: RegExp[];
  private tablePatterns: RegExp[];
  private typePatterns: RegExp[];
  private effectPatterns: Map<string, RegExp[]>;

  constructor() {
    // Route patterns
    this.routePatterns = [
      /\/\w+/g,                    // /users, /api, etc.
      /\b\w+\s*endpoint\b/gi,      // "user endpoint"
      /\b\w+\s*route\b/gi,         // "api route"
      /\b\w+\s*path\b/gi,          // "user path"
    ];

    // Table patterns
    this.tablePatterns = [
      /\b\w+\s*table\b/gi,         // "user table"
      /\bfrom\s+(\w+)\b/gi,        // "from users"
      /\binto\s+(\w+)\b/gi,        // "into orders"
      /\bupdate\s+(\w+)\b/gi,      // "update products"
    ];

    // Type patterns
    this.typePatterns = [
      /\b[A-Z]\w*Type\b/g,         // UserType, OrderType
      /\b[A-Z]\w*Interface\b/g,    // UserInterface
      /\b[A-Z]\w*Schema\b/g,       // UserSchema
      /\binterface\s+(\w+)\b/gi,   // "interface User"
      /\btype\s+(\w+)\b/gi,        // "type User"
    ];

    // Effect patterns
    this.effectPatterns = new Map([
      ['db', [/\bdatabase\b/gi, /\bdb\b/gi, /\bquery\b/gi, /\bsql\b/gi]],
      ['fs', [/\bfile\b/gi, /\bread\b/gi, /\bwrite\b/gi, /\bsave\b/gi]],
      ['net', [/\bhttp\b/gi, /\bapi\b/gi, /\brequest\b/gi, /\bfetch\b/gi]],
      ['auth', [/\bauth\b/gi, /\blogin\b/gi, /\btoken\b/gi, /\bpermission\b/gi]],
      ['cache', [/\bcache\b/gi, /\bredis\b/gi, /\bmemory\b/gi]],
      ['email', [/\bemail\b/gi, /\bmail\b/gi, /\bsmtp\b/gi]],
      ['crypto', [/\bencrypt\b/gi, /\bhash\b/gi, /\bsign\b/gi, /\bcrypto\b/gi]]
    ]);
  }

  extractFeatures(queryText: string): QueryFeatures {
    const features: QueryFeatures = {
      routes: new Set(),
      tables: new Set(),
      types: new Set(),
      tokens: new Set(),
      effects: new Set()
    };

    const lowerQuery = queryText.toLowerCase();

    // Extract routes
    for (const pattern of this.routePatterns) {
      const matches = queryText.matchAll(pattern);
      for (const match of matches) {
        if (match[1]) {
          features.routes.add(match[1]);
        } else if (match[0]) {
          features.routes.add(match[0].trim());
        }
      }
    }

    // Extract tables
    for (const pattern of this.tablePatterns) {
      const matches = queryText.matchAll(pattern);
      for (const match of matches) {
        if (match[1]) {
          features.tables.add(match[1].toLowerCase());
        } else {
          const words = match[0].split(/\s+/);
          features.tables.add(words[0].toLowerCase());
        }
      }
    }

    // Extract types
    for (const pattern of this.typePatterns) {
      const matches = queryText.matchAll(pattern);
      for (const match of matches) {
        if (match[1]) {
          features.types.add(match[1]);
        } else {
          features.types.add(match[0].trim());
        }
      }
    }

    // Extract effects
    for (const [effect, patterns] of this.effectPatterns) {
      for (const pattern of patterns) {
        if (pattern.test(lowerQuery)) {
          features.effects.add(effect);
          break;
        }
      }
    }

    // Extract domain tokens (significant words)
    const words = queryText
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(word => 
        word.length > 3 && 
        !this.isStopWord(word)
      );
    
    for (const word of words) {
      features.tokens.add(word);
    }

    return features;
  }

  private isStopWord(word: string): boolean {
    const stopWords = new Set([
      'the', 'and', 'for', 'with', 'from', 'that', 'this', 'have', 'need',
      'want', 'find', 'show', 'get', 'create', 'make', 'use', 'can', 'will',
      'should', 'would', 'could', 'how', 'what', 'where', 'when', 'why'
    ]);
    
    return stopWords.has(word);
  }
}

/**
 * Main runtime feature computer
 */
export class RaptorRuntimeFeatures {
  private embeddingService: RaptorEmbeddingService;
  private queryExtractor: QueryFeatureExtractor;
  private snapshots: Map<string, RaptorSnapshot>;
  private semanticCards: Map<string, SemanticCard>;
  private config: FeatureComputationConfig;

  constructor(
    embeddingService: RaptorEmbeddingService,
    config?: Partial<FeatureComputationConfig>
  ) {
    this.embeddingService = embeddingService;
    this.queryExtractor = new QueryFeatureExtractor();
    this.snapshots = new Map();
    this.semanticCards = new Map();
    this.config = {
      enabled: true,
      nlThreshold: 0.3,
      maxCandidates: 1000,
      priorBoostCap: 0.5,
      topicThreshold: 0.35,
      ...config
    };
  }

  loadSnapshot(repoSha: string, snapshot: RaptorSnapshot): void {
    this.snapshots.set(repoSha, snapshot);
  }

  loadSemanticCard(fileId: string, card: SemanticCard): void {
    this.semanticCards.set(fileId, card);
  }

  updateConfig(newConfig: Partial<FeatureComputationConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  async prepareQuery(queryText: string): Promise<QueryEmbedding> {
    const features = this.queryExtractor.extractFeatures(queryText);
    const semantic = await this.embeddingService.embedSummary(queryText);

    return { semantic, features };
  }

  async computeFeatures(
    repoSha: string,
    candidateFileIds: string[],
    queryEmbedding: QueryEmbedding,
    nlScore?: number
  ): Promise<Map<string, RaptorFeatures>> {
    const features = new Map<string, RaptorFeatures>();

    if (!this.shouldComputeFeatures(candidateFileIds.length, nlScore)) {
      // Return default features for all candidates
      for (const fileId of candidateFileIds) {
        features.set(fileId, this.getDefaultFeatures());
      }
      return features;
    }

    const snapshot = this.snapshots.get(repoSha);
    if (!snapshot) {
      // No snapshot available, return default features
      for (const fileId of candidateFileIds) {
        features.set(fileId, this.getDefaultFeatures());
      }
      return features;
    }

    for (const fileId of candidateFileIds) {
      const fileFeatures = await this.computeFileFeatures(
        fileId,
        snapshot,
        queryEmbedding
      );
      features.set(fileId, fileFeatures);
    }

    return features;
  }

  async computePathPriorBoosts(
    repoSha: string,
    candidateFileIds: string[],
    queryEmbedding: QueryEmbedding
  ): Promise<PathPriorBoost[]> {
    const boosts: PathPriorBoost[] = [];

    if (!this.config.enabled) {
      return boosts;
    }

    const snapshot = this.snapshots.get(repoSha);
    if (!snapshot) {
      return boosts;
    }

    // Group files by their leaf nodes
    const nodeToFiles = new Map<string, string[]>();
    
    for (const fileId of candidateFileIds) {
      const leafNodes = snapshot.fileToLeaf.get(fileId) || [];
      for (const nodeId of leafNodes) {
        if (!nodeToFiles.has(nodeId)) {
          nodeToFiles.set(nodeId, []);
        }
        nodeToFiles.get(nodeId)!.push(fileId);
      }
    }

    // Compute similarity for each node and its ancestors
    for (const [nodeId, files] of nodeToFiles) {
      const ancestors = this.getAncestors(snapshot, nodeId);
      let maxSim = 0;
      let bestNode: RaptorNode | null = null;

      for (const ancestor of ancestors) {
        if (ancestor.summaryEmb.length === 0) continue;
        
        const similarity = this.embeddingService.cosineSimilarity(
          queryEmbedding.semantic,
          ancestor.summaryEmb
        );

        if (similarity > maxSim) {
          maxSim = similarity;
          bestNode = ancestor;
        }
      }

      // Apply boost if similarity exceeds threshold
      if (maxSim > this.config.topicThreshold && bestNode) {
        const boost = Math.min(
          this.config.priorBoostCap,
          (maxSim - this.config.topicThreshold) * 2 // Scale to [0, priorBoostCap]
        );

        for (const fileId of files) {
          boosts.push({
            fileId,
            boost,
            reason: `Matched node ${bestNode.id} (sim=${maxSim.toFixed(3)})`
          });
        }
      }
    }

    return boosts;
  }

  private shouldComputeFeatures(candidateCount: number, nlScore?: number): boolean {
    if (!this.config.enabled) return false;
    
    // Compute if NL threshold passed OR candidate count is low
    return (nlScore !== undefined && nlScore >= this.config.nlThreshold) ||
           candidateCount < this.config.maxCandidates;
  }

  private async computeFileFeatures(
    fileId: string,
    snapshot: RaptorSnapshot,
    queryEmbedding: QueryEmbedding
  ): Promise<RaptorFeatures> {
    const card = this.semanticCards.get(fileId);
    if (!card) {
      return this.getDefaultFeatures();
    }

    // Get ancestors for this file
    const leafNodeIds = snapshot.fileToLeaf.get(fileId) || [];
    const ancestors = leafNodeIds.flatMap(nodeId => this.getAncestors(snapshot, nodeId));

    // Compute raptor_max_sim and raptor_depth_of_best
    let maxSim = 0;
    let depthOfBest = 1;

    for (const ancestor of ancestors) {
      if (ancestor.summaryEmb.length === 0) continue;

      const similarity = this.embeddingService.cosineSimilarity(
        queryEmbedding.semantic,
        ancestor.summaryEmb
      );

      if (similarity > maxSim) {
        maxSim = similarity;
        depthOfBest = ancestor.level + 1; // Convert 0-indexed to 1-indexed
      }
    }

    // Compute topic_overlap (Jaccard similarity)
    const topicOverlap = this.computeTopicOverlap(queryEmbedding.features, card);

    return {
      raptor_max_sim: maxSim,
      raptor_depth_of_best: depthOfBest,
      topic_overlap: topicOverlap,
      B: card.B
    };
  }

  private getAncestors(snapshot: RaptorSnapshot, nodeId: string): RaptorNode[] {
    const ancestors: RaptorNode[] = [];
    let current = snapshot.nodes.get(nodeId);

    while (current) {
      ancestors.push(current);
      current = current.parent ? snapshot.nodes.get(current.parent) : undefined;
    }

    return ancestors;
  }

  private computeTopicOverlap(queryFeatures: QueryFeatures, card: SemanticCard): number {
    // Compute Jaccard similarity between query features and card features
    const querySet = new Set<string>();
    
    // Add query features to set
    for (const route of queryFeatures.routes) querySet.add(`route:${route}`);
    for (const table of queryFeatures.tables) querySet.add(`table:${table}`);
    for (const type of queryFeatures.types) querySet.add(`type:${type}`);
    for (const token of queryFeatures.tokens) querySet.add(`token:${token}`);
    for (const effect of queryFeatures.effects) querySet.add(`effect:${effect}`);

    const cardSet = new Set<string>();
    
    // Add card features to set
    for (const route of card.resources.routes) cardSet.add(`route:${route}`);
    for (const table of card.resources.sql) cardSet.add(`table:${table}`);
    for (const type of card.shapes.typeNames) cardSet.add(`type:${type}`);
    for (const token of card.domainTokens) cardSet.add(`token:${token}`);
    for (const effect of card.effects) cardSet.add(`effect:${effect}`);

    // Compute Jaccard similarity
    const intersection = new Set([...querySet].filter(x => cardSet.has(x)));
    const union = new Set([...querySet, ...cardSet]);

    return union.size > 0 ? intersection.size / union.size : 0;
  }

  private getDefaultFeatures(): RaptorFeatures {
    return {
      raptor_max_sim: 0,
      raptor_depth_of_best: 1,
      topic_overlap: 0,
      B: 0
    };
  }

  // Debugging methods
  getFeatureDebugInfo(fileId: string, repoSha: string): any {
    const snapshot = this.snapshots.get(repoSha);
    const card = this.semanticCards.get(fileId);
    
    if (!snapshot || !card) {
      return { error: 'Snapshot or card not found' };
    }

    const leafNodeIds = snapshot.fileToLeaf.get(fileId) || [];
    const ancestors = leafNodeIds.flatMap(nodeId => this.getAncestors(snapshot, nodeId));

    return {
      fileId,
      card: {
        roles: card.roles,
        resourceCount: Object.values(card.resources).reduce((sum, arr) => sum + arr.length, 0),
        typeCount: card.shapes.typeNames.length,
        domainTokenCount: card.domainTokens.length,
        B: card.B
      },
      ancestors: ancestors.map(node => ({
        id: node.id,
        level: node.level,
        summary: node.summary.substring(0, 100) + '...',
        pressure: node.pressure
      })),
      config: this.config
    };
  }

  clearCache(): void {
    this.snapshots.clear();
    this.semanticCards.clear();
  }
}

export default RaptorRuntimeFeatures;