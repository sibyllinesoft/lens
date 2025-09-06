/**
 * Cross-Repo Moniker Linking System
 * 
 * Ingests LSIF "monikers"/package metadata to build global symbol-ID space.
 * During Stage-B/B++ lifts hits into moniker clusters and imports spans from downstream repos.
 * Uses centrality prior topic-normalized across repos to avoid over-favoring ecosystem hubs.
 * Caches small "import cones" per package version.
 * 
 * Gates: Recall@50 +0.6–1.0 pp on cross-repo queries, p95 ≤ +0.6 ms, why-mix KL ≤ 0.02
 * Constraints: cluster≤128, import-cone depth≤1, enable for {symbol,NL} intents
 */

import { LensTracer } from '../telemetry/tracer.js';
import type { SearchHit, SymbolCandidate, MatchReason } from './span_resolver/types.js';

export interface LSIFMoniker {
  scheme: string;              // npm, pypi, maven, etc.
  identifier: string;          // package@version::symbol
  kind: 'import' | 'export' | 'local';
  unique: 'document' | 'project' | 'scheme' | 'global';
}

export interface MonikerCluster {
  id: string;                  // Global cluster ID
  canonical_identifier: string; // Primary symbol identifier
  package_info: {
    name: string;
    version: string;
    scheme: string;            // npm, pypi, etc.
  };
  symbols: MonikerSymbol[];
  centrality_score: number;    // Topic-normalized centrality
  import_cone: ImportCone;     // Cached downstream references
}

export interface MonikerSymbol {
  identifier: string;
  kind: string;                // function, class, type, etc.
  signature?: string;
  documentation?: string;
  locations: SymbolLocation[];
  cross_repo_refs: CrossRepoReference[];
}

export interface SymbolLocation {
  repo_sha: string;
  file_path: string;
  line: number;
  col: number;
  span_len?: number;
}

export interface CrossRepoReference {
  source_repo: string;
  target_repo: string;
  confidence: number;          // 0.0-1.0
  ref_type: 'import' | 'extends' | 'implements' | 'calls';
}

export interface ImportCone {
  depth: number;               // ≤1 per constraints
  downstream_refs: DownstreamRef[];
  cached_at: Date;
  cache_ttl: number;           // TTL in seconds
}

export interface DownstreamRef {
  repo_sha: string;
  symbols: string[];
  usage_frequency: number;
  last_seen: Date;
}

export interface MonikerLinkingConfig {
  max_cluster_size: number;    // ≤128 per constraints
  max_import_cone_depth: number; // ≤1 per constraints
  cache_ttl_hours: number;
  centrality_normalization_factor: number;
  supported_intents: ('symbol' | 'nl')[];
  performance_targets: {
    max_additional_latency_ms: number; // ≤0.6ms
    min_recall_improvement_pp: number; // ≥0.6pp
    max_why_mix_kl_divergence: number; // ≤0.02
  };
}

export class MonikerLinkingSystem {
  private clusters = new Map<string, MonikerCluster>();
  private symbolToCluster = new Map<string, string>();
  private importConeCache = new Map<string, ImportCone>();
  
  constructor(
    private config: MonikerLinkingConfig = {
      max_cluster_size: 128,
      max_import_cone_depth: 1,
      cache_ttl_hours: 24,
      centrality_normalization_factor: 0.85,
      supported_intents: ['symbol', 'nl'],
      performance_targets: {
        max_additional_latency_ms: 0.6,
        min_recall_improvement_pp: 0.6,
        max_why_mix_kl_divergence: 0.02,
      }
    }
  ) {}

  /**
   * Ingest LSIF moniker data to build global symbol space
   */
  async ingestLSIFMonikers(
    repoSha: string, 
    lsifData: LSIFMoniker[], 
    packageMetadata: any
  ): Promise<void> {
    const span = LensTracer.createChildSpan('ingest_lsif_monikers', {
      'repo.sha': repoSha,
      'monikers.count': lsifData.length,
    });

    try {
      for (const moniker of lsifData) {
        await this.processMoniker(repoSha, moniker, packageMetadata);
      }

      // Update centrality scores after ingestion
      await this.updateCentralityScores();
      
      span.setAttributes({
        'clusters.created': this.clusters.size,
        success: true
      });

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Process individual LSIF moniker into cluster system
   */
  private async processMoniker(
    repoSha: string, 
    moniker: LSIFMoniker, 
    metadata: any
  ): Promise<void> {
    const clusterId = this.generateClusterId(moniker);
    
    let cluster = this.clusters.get(clusterId);
    if (!cluster) {
      cluster = {
        id: clusterId,
        canonical_identifier: moniker.identifier,
        package_info: this.extractPackageInfo(moniker, metadata),
        symbols: [],
        centrality_score: 0.0,
        import_cone: {
          depth: 0,
          downstream_refs: [],
          cached_at: new Date(),
          cache_ttl: this.config.cache_ttl_hours * 3600
        }
      };
      this.clusters.set(clusterId, cluster);
    }

    // Enforce cluster size constraint
    if (cluster.symbols.length >= this.config.max_cluster_size) {
      return; // Skip to maintain ≤128 constraint
    }

    const symbol: MonikerSymbol = {
      identifier: moniker.identifier,
      kind: metadata.kind || 'unknown',
      signature: metadata.signature,
      documentation: metadata.documentation,
      locations: [{
        repo_sha: repoSha,
        file_path: metadata.file_path,
        line: metadata.line,
        col: metadata.col,
        span_len: metadata.span_len
      }],
      cross_repo_refs: []
    };

    cluster.symbols.push(symbol);
    this.symbolToCluster.set(moniker.identifier, clusterId);
  }

  /**
   * Lift Stage-B hits into moniker clusters during query processing
   */
  async expandWithMonikerClusters(
    candidates: SymbolCandidate[],
    queryIntent: string
  ): Promise<SymbolCandidate[]> {
    // Only process supported intents
    if (!this.config.supported_intents.includes(queryIntent as any)) {
      return candidates;
    }

    const span = LensTracer.createChildSpan('expand_moniker_clusters', {
      'candidates.input': candidates.length,
      'query.intent': queryIntent
    });

    const startTime = performance.now();
    const expandedCandidates: SymbolCandidate[] = [...candidates];

    try {
      for (const candidate of candidates) {
        const clusterId = this.findClusterForCandidate(candidate);
        if (clusterId) {
          const clusterCandidates = await this.expandFromCluster(
            clusterId, 
            candidate,
            queryIntent
          );
          expandedCandidates.push(...clusterCandidates);
        }
      }

      // Apply caps and vendor veto enforcement
      const vetted = this.applyVendorVeto(expandedCandidates);
      const capped = this.applyCentralityPrior(vetted);

      const elapsedMs = performance.now() - startTime;
      
      span.setAttributes({
        'candidates.output': capped.length,
        'candidates.expanded': capped.length - candidates.length,
        'latency.ms': elapsedMs,
        'latency.within_budget': elapsedMs <= this.config.performance_targets.max_additional_latency_ms,
        success: true
      });

      // Verify performance constraint
      if (elapsedMs > this.config.performance_targets.max_additional_latency_ms) {
        console.warn(`Moniker expansion exceeded latency budget: ${elapsedMs}ms > ${this.config.performance_targets.max_additional_latency_ms}ms`);
      }

      return capped;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Find cluster for a given candidate symbol
   */
  private findClusterForCandidate(candidate: SymbolCandidate): string | null {
    // Try direct symbol lookup first
    const directMatch = this.symbolToCluster.get(candidate.file_path);
    if (directMatch) return directMatch;

    // Fall back to fuzzy matching based on file path and symbol kind
    for (const [clusterId, cluster] of this.clusters) {
      for (const symbol of cluster.symbols) {
        if (this.isSymbolMatch(candidate, symbol)) {
          return clusterId;
        }
      }
    }

    return null;
  }

  /**
   * Expand candidates from a moniker cluster
   */
  private async expandFromCluster(
    clusterId: string,
    sourceCandidate: SymbolCandidate,
    queryIntent: string
  ): Promise<SymbolCandidate[]> {
    const cluster = this.clusters.get(clusterId);
    if (!cluster) return [];

    const expanded: SymbolCandidate[] = [];

    // Add symbols from the cluster
    for (const symbol of cluster.symbols) {
      for (const location of symbol.locations) {
        // Skip if same as source candidate
        if (location.file_path === sourceCandidate.file_path) continue;

        const candidate: SymbolCandidate = {
          file_path: location.file_path,
          score: sourceCandidate.score * cluster.centrality_score,
          match_reasons: [...sourceCandidate.match_reasons, 'symbol'] as MatchReason[],
          symbol_kind: symbol.kind as any,
          ast_path: symbol.signature,
          upstream_line: location.line,
          upstream_col: location.col
        };

        expanded.push(candidate);
      }
    }

    // Import spans from downstream repos via cached import cones
    const importCone = await this.getImportCone(clusterId);
    if (importCone.depth <= this.config.max_import_cone_depth) {
      for (const downstreamRef of importCone.downstream_refs) {
        const downstreamCandidates = await this.expandFromDownstream(
          downstreamRef, 
          sourceCandidate,
          cluster
        );
        expanded.push(...downstreamCandidates);
      }
    }

    return expanded;
  }

  /**
   * Get cached import cone for cluster, refreshing if needed
   */
  private async getImportCone(clusterId: string): Promise<ImportCone> {
    const cached = this.importConeCache.get(clusterId);
    const cluster = this.clusters.get(clusterId);
    
    if (!cluster) {
      return { depth: 0, downstream_refs: [], cached_at: new Date(), cache_ttl: 0 };
    }

    if (cached && this.isCacheValid(cached)) {
      return cached;
    }

    // Rebuild import cone
    const importCone = await this.buildImportCone(cluster);
    this.importConeCache.set(clusterId, importCone);
    return importCone;
  }

  /**
   * Build import cone for cluster (depth ≤ 1)
   */
  private async buildImportCone(cluster: MonikerCluster): Promise<ImportCone> {
    const downstreamRefs: DownstreamRef[] = [];

    // Depth 1: direct imports only (per constraint)
    for (const symbol of cluster.symbols) {
      for (const crossRef of symbol.cross_repo_refs) {
        if (crossRef.ref_type === 'import' && crossRef.confidence > 0.7) {
          const existing = downstreamRefs.find(ref => ref.repo_sha === crossRef.target_repo);
          if (existing) {
            existing.symbols.push(symbol.identifier);
            existing.usage_frequency++;
          } else {
            downstreamRefs.push({
              repo_sha: crossRef.target_repo,
              symbols: [symbol.identifier],
              usage_frequency: 1,
              last_seen: new Date()
            });
          }
        }
      }
    }

    return {
      depth: 1, // Enforces ≤1 constraint
      downstream_refs: downstreamRefs,
      cached_at: new Date(),
      cache_ttl: this.config.cache_ttl_hours * 3600
    };
  }

  /**
   * Apply vendor veto enforcement
   */
  private applyVendorVeto(candidates: SymbolCandidate[]): SymbolCandidate[] {
    // Implement vendor veto logic - prioritize local/internal packages over external
    return candidates.filter(candidate => {
      // For now, basic implementation - can be enhanced with vendor rules
      return true;
    });
  }

  /**
   * Apply centrality prior topic-normalized across repos
   */
  private applyCentralityPrior(candidates: SymbolCandidate[]): SymbolCandidate[] {
    const normalized = candidates.map(candidate => {
      const clusterId = this.findClusterForCandidate(candidate);
      if (clusterId) {
        const cluster = this.clusters.get(clusterId);
        if (cluster) {
          // Apply topic-normalized centrality score
          candidate.score *= cluster.centrality_score * this.config.centrality_normalization_factor;
        }
      }
      return candidate;
    });

    // Sort by adjusted score
    return normalized.sort((a, b) => b.score - a.score);
  }

  /**
   * Update centrality scores for all clusters
   */
  private async updateCentralityScores(): Promise<void> {
    for (const [clusterId, cluster] of this.clusters) {
      cluster.centrality_score = this.calculateCentralityScore(cluster);
    }
  }

  /**
   * Calculate centrality score for a cluster
   */
  private calculateCentralityScore(cluster: MonikerCluster): number {
    // Basic centrality calculation - can be enhanced with PageRank-like algorithm
    const symbolCount = cluster.symbols.length;
    const locationCount = cluster.symbols.reduce((acc, s) => acc + s.locations.length, 0);
    const crossRepoCount = cluster.symbols.reduce((acc, s) => acc + s.cross_repo_refs.length, 0);
    
    // Normalize to avoid over-favoring ecosystem hubs
    const rawScore = (symbolCount * 0.3) + (locationCount * 0.4) + (crossRepoCount * 0.3);
    return Math.min(1.0, rawScore / 100.0); // Cap at 1.0
  }

  // Helper methods
  private generateClusterId(moniker: LSIFMoniker): string {
    return `${moniker.scheme}:${moniker.identifier}`;
  }

  private extractPackageInfo(moniker: LSIFMoniker, metadata: any) {
    const parts = moniker.identifier.split('::');
    const packagePart = parts[0];
    const [name, version] = packagePart.split('@');
    
    return {
      name: name || 'unknown',
      version: version || '0.0.0',
      scheme: moniker.scheme
    };
  }

  private isSymbolMatch(candidate: SymbolCandidate, symbol: MonikerSymbol): boolean {
    return symbol.locations.some(loc => 
      loc.file_path === candidate.file_path ||
      (candidate.symbol_kind === symbol.kind && candidate.ast_path?.includes(symbol.identifier))
    );
  }

  private async expandFromDownstream(
    downstreamRef: DownstreamRef,
    sourceCandidate: SymbolCandidate,
    cluster: MonikerCluster
  ): Promise<SymbolCandidate[]> {
    // Placeholder for downstream expansion logic
    // In production, this would query the downstream repo's index
    return [];
  }

  private isCacheValid(importCone: ImportCone): boolean {
    const ageSeconds = (Date.now() - importCone.cached_at.getTime()) / 1000;
    return ageSeconds < importCone.cache_ttl;
  }

  /**
   * Get performance metrics for monitoring gates
   */
  getPerformanceMetrics() {
    return {
      clusters_count: this.clusters.size,
      symbols_indexed: Array.from(this.clusters.values()).reduce((acc, c) => acc + c.symbols.length, 0),
      import_cones_cached: this.importConeCache.size,
      avg_centrality_score: this.getAverageCentralityScore(),
    };
  }

  private getAverageCentralityScore(): number {
    const scores = Array.from(this.clusters.values()).map(c => c.centrality_score);
    return scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
  }
}