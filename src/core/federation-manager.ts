/**
 * Multi-Repository Federation Manager
 * Enables searching across multiple repositories with intelligent routing and result merging
 * Implements distributed search coordination with conflict resolution and ranking normalization
 */

import { createHash } from 'crypto';
import { performance } from 'perf_hooks';
import { LensTracer } from '../telemetry/tracer.js';
import { globalCacheManager } from './advanced-cache-manager.js';
import { globalParallelProcessor } from './parallel-processor.js';
import type { SearchContext, SearchHit, Candidate } from '../types/core.js';

interface FederatedRepository {
  id: string;
  name: string;
  url: string;
  priority: number;
  weight: number;
  isLocal: boolean;
  isActive: boolean;
  healthScore: number;
  lastSync: number;
  searchEndpoint: string;
  indexVersion: string;
  metadata: {
    languages: string[];
    fileCount: number;
    lastCommit: string;
    description?: string;
    tags: string[];
  };
}

interface FederationConfig {
  maxConcurrentRepos: number;
  searchTimeout: number;
  resultMergingStrategy: 'round_robin' | 'score_weighted' | 'priority_based';
  scoreNormalization: boolean;
  crossRepoDeduplication: boolean;
  federatedCaching: boolean;
  healthCheckInterval: number;
}

interface SearchRoute {
  repositories: FederatedRepository[];
  strategy: SearchStrategy;
  expectedLatency: number;
  confidenceScore: number;
}

interface FederatedSearchResult {
  hits: SearchHit[];
  repositoryResults: Map<string, {
    hits: SearchHit[];
    latency: number;
    status: 'success' | 'timeout' | 'error';
    error?: string;
  }>;
  totalLatency: number;
  strategy: SearchStrategy;
  mergedCount: number;
  deduplicatedCount: number;
}

enum SearchStrategy {
  SCATTER_GATHER = 'scatter_gather',    // Search all repos in parallel
  PRIORITY_FIRST = 'priority_first',    // Search high-priority repos first
  ADAPTIVE_ROUTING = 'adaptive_routing', // Route based on query characteristics
  LOCALITY_AWARE = 'locality_aware',    // Prefer local repos
  LOAD_BALANCED = 'load_balanced'       // Balance load across repos
}

interface RepositoryHealth {
  repoId: string;
  isHealthy: boolean;
  responseTime: number;
  successRate: number;
  errorCount: number;
  lastHealthCheck: number;
  issues: string[];
}

export class FederationManager {
  private static instance: FederationManager;
  
  // Repository management
  private repositories: Map<string, FederatedRepository> = new Map();
  private repositoryHealth: Map<string, RepositoryHealth> = new Map();
  private routingTable: Map<string, SearchRoute[]> = new Map();
  
  // Configuration
  private config: FederationConfig;
  
  // Performance tracking
  private searchStats = {
    totalSearches: 0,
    federatedSearches: 0,
    avgLatency: 0,
    successRate: 0,
    crossRepoHits: 0
  };
  
  // Timers
  private healthCheckTimer?: NodeJS.Timeout;
  private routingOptimizationTimer?: NodeJS.Timeout;
  
  private constructor() {
    this.config = {
      maxConcurrentRepos: 10,
      searchTimeout: 15000, // 15 seconds
      resultMergingStrategy: 'score_weighted',
      scoreNormalization: true,
      crossRepoDeduplication: true,
      federatedCaching: true,
      healthCheckInterval: 60000 // 1 minute
    };
    
    this.startMaintenanceTimers();
  }
  
  public static getInstance(): FederationManager {
    if (!FederationManager.instance) {
      FederationManager.instance = new FederationManager();
    }
    return FederationManager.instance;
  }
  
  /**
   * Register a repository for federated search
   */
  async registerRepository(repo: Omit<FederatedRepository, 'id' | 'healthScore' | 'lastSync'>): Promise<string> {
    const span = LensTracer.createChildSpan('register_federated_repo');
    
    try {
      const repoId = this.generateRepositoryId(repo.name, repo.url);
      
      const federatedRepo: FederatedRepository = {
        id: repoId,
        healthScore: 1.0,
        lastSync: Date.now(),
        ...repo
      };
      
      this.repositories.set(repoId, federatedRepo);
      
      // Initialize health tracking
      this.repositoryHealth.set(repoId, {
        repoId,
        isHealthy: true,
        responseTime: 0,
        successRate: 1.0,
        errorCount: 0,
        lastHealthCheck: Date.now(),
        issues: []
      });
      
      // Perform initial health check
      await this.performHealthCheck(repoId);
      
      // Update routing table
      this.updateRoutingTable();
      
      console.log(`📡 Registered federated repository: ${repo.name} (${repoId})`);
      
      span.setAttributes({
        success: true,
        repo_id: repoId,
        repo_name: repo.name,
        is_local: repo.isLocal,
        priority: repo.priority
      });
      
      return repoId;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Unregister a repository
   */
  async unregisterRepository(repoId: string): Promise<void> {
    const span = LensTracer.createChildSpan('unregister_federated_repo');
    
    try {
      const repo = this.repositories.get(repoId);
      if (!repo) {
        throw new Error(`Repository ${repoId} not found`);
      }
      
      this.repositories.delete(repoId);
      this.repositoryHealth.delete(repoId);
      
      // Update routing table
      this.updateRoutingTable();
      
      console.log(`📡 Unregistered federated repository: ${repo.name} (${repoId})`);
      
      span.setAttributes({
        success: true,
        repo_id: repoId,
        repo_name: repo.name
      });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Perform federated search across multiple repositories
   */
  async federatedSearch(context: SearchContext): Promise<FederatedSearchResult> {
    const span = LensTracer.createSearchSpan(context);
    const startTime = performance.now();
    this.searchStats.totalSearches++;
    
    try {
      // Check federated cache first
      if (this.config.federatedCaching) {
        const cacheKey = this.generateFederatedCacheKey(context);
        const cached = await globalCacheManager.get<FederatedSearchResult>(cacheKey, context);
        if (cached) {
          return cached;
        }
      }
      
      // Determine optimal search strategy and repositories
      const searchRoute = this.selectOptimalSearchRoute(context);
      
      if (searchRoute.repositories.length === 0) {
        throw new Error('No healthy repositories available for federated search');
      }
      
      this.searchStats.federatedSearches++;
      
      // Execute search across selected repositories
      const searchPromises = searchRoute.repositories.map(repo => 
        this.searchRepository(repo, context)
      );
      
      // Wait for all searches with timeout
      const searchResults = await Promise.allSettled(
        searchPromises.map(p => this.withTimeout(p, this.config.searchTimeout))
      );
      
      // Process and merge results
      const repositoryResults = new Map<string, any>();
      const allHits: SearchHit[] = [];
      
      for (let i = 0; i < searchResults.length; i++) {
        const result = searchResults[i];
        const repo = searchRoute.repositories[i];
        
        if (result.status === 'fulfilled') {
          const repoResult = result.value;
          repositoryResults.set(repo.id, {
            hits: repoResult.hits,
            latency: repoResult.latency,
            status: 'success' as const
          });
          
          // Add repository context to hits
          const enhancedHits = repoResult.hits.map((hit: SearchHit) => ({
            ...hit,
            repository: {
              id: repo.id,
              name: repo.name,
              priority: repo.priority,
              weight: repo.weight
            }
          }));
          
          allHits.push(...enhancedHits);
          
        } else {
          const error = result.reason instanceof Error ? result.reason.message : 'Unknown error';
          repositoryResults.set(repo.id, {
            hits: [],
            latency: 0,
            status: 'error' as const,
            error
          });
          
          // Update health score
          this.updateRepositoryHealth(repo.id, false, error);
        }
      }
      
      // Merge and deduplicate results
      const mergedHits = this.mergeSearchResults(allHits, searchRoute.strategy);
      const finalHits = this.config.crossRepoDeduplication ? 
        this.deduplicateResults(mergedHits) : mergedHits;
      
      const totalLatency = performance.now() - startTime;
      
      const federatedResult: FederatedSearchResult = {
        hits: finalHits.slice(0, context.k),
        repositoryResults,
        totalLatency,
        strategy: searchRoute.strategy,
        mergedCount: allHits.length,
        deduplicatedCount: mergedHits.length - finalHits.length
      };
      
      // Cache result
      if (this.config.federatedCaching) {
        const cacheKey = this.generateFederatedCacheKey(context);
        await globalCacheManager.set(cacheKey, federatedResult, 300000, context); // 5 minute TTL
      }
      
      // Update statistics
      this.updateSearchStats(totalLatency, repositoryResults.size > 1);
      
      span.setAttributes({
        success: true,
        repositories_searched: searchRoute.repositories.length,
        total_hits: allHits.length,
        final_hits: finalHits.length,
        strategy: searchRoute.strategy,
        latency_ms: totalLatency
      });
      
      return federatedResult;
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Search individual repository
   */
  private async searchRepository(repo: FederatedRepository, context: SearchContext): Promise<{
    hits: SearchHit[];
    latency: number;
  }> {
    const span = LensTracer.createChildSpan('search_federated_repo');
    const startTime = performance.now();
    
    try {
      let hits: SearchHit[] = [];
      
      if (repo.isLocal) {
        // Use local search engine for local repositories
        hits = await this.searchLocalRepository(repo, context);
      } else {
        // Use HTTP API for remote repositories
        hits = await this.searchRemoteRepository(repo, context);
      }
      
      const latency = performance.now() - startTime;
      
      // Update repository health
      this.updateRepositoryHealth(repo.id, true);
      
      span.setAttributes({
        success: true,
        repo_id: repo.id,
        repo_name: repo.name,
        hits_count: hits.length,
        latency_ms: latency,
        is_local: repo.isLocal
      });
      
      return { hits, latency };
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ 
        success: false, 
        error: errorMsg,
        repo_id: repo.id,
        repo_name: repo.name
      });
      
      // Update repository health
      this.updateRepositoryHealth(repo.id, false, errorMsg);
      
      throw error;
    } finally {
      span.end();
    }
  }
  
  /**
   * Search local repository using parallel processor
   */
  private async searchLocalRepository(repo: FederatedRepository, context: SearchContext): Promise<SearchHit[]> {
    // Create a modified context for the specific repository
    const repoContext: SearchContext = {
      ...context,
      repo_sha: repo.id, // Use repo ID as repo_sha for local search
    };
    
    // Use parallel processor to execute search
    const result = await globalParallelProcessor.submitTask(
      'lexical_search' as any, // TaskType from parallel-processor
      { context: repoContext },
      0, // CRITICAL priority
      repoContext,
      10000 // 10 second timeout
    );
    
    return result.hits || [];
  }
  
  /**
   * Search remote repository via HTTP API
   */
  private async searchRemoteRepository(repo: FederatedRepository, context: SearchContext): Promise<SearchHit[]> {
    const searchUrl = `${repo.searchEndpoint}/search`;
    
    const requestBody = {
      q: context.query,
      mode: context.mode,
      k: Math.min(context.k, 100), // Limit remote requests
      fuzzy_distance: context.fuzzy_distance,
      repo_sha: repo.id
    };
    
    const response = await fetch(searchUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'LensFederation/1.0'
      },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(this.config.searchTimeout)
    });
    
    if (!response.ok) {
      throw new Error(`Remote search failed: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    return data.hits || [];
  }
  
  /**
   * Select optimal search route based on query and repository characteristics
   */
  private selectOptimalSearchRoute(context: SearchContext): SearchRoute {
    const availableRepos = Array.from(this.repositories.values())
      .filter(repo => repo.isActive)
      .filter(repo => this.isRepositoryHealthy(repo.id));
    
    if (availableRepos.length === 0) {
      return {
        repositories: [],
        strategy: SearchStrategy.SCATTER_GATHER,
        expectedLatency: 0,
        confidenceScore: 0
      };
    }
    
    // Analyze query characteristics
    const queryFeatures = this.analyzeQueryFeatures(context);
    
    // Select strategy based on query and repository characteristics
    let strategy: SearchStrategy;
    let selectedRepos: FederatedRepository[];
    
    if (queryFeatures.isSpecific && availableRepos.some(r => r.isLocal)) {
      // For specific queries, prefer local repositories
      strategy = SearchStrategy.LOCALITY_AWARE;
      selectedRepos = this.selectLocalityAwareRepos(availableRepos, queryFeatures);
    } else if (queryFeatures.isComplex) {
      // For complex queries, use adaptive routing
      strategy = SearchStrategy.ADAPTIVE_ROUTING;
      selectedRepos = this.selectAdaptiveRepos(availableRepos, queryFeatures);
    } else if (availableRepos.length > this.config.maxConcurrentRepos) {
      // For many repos, use priority-based selection
      strategy = SearchStrategy.PRIORITY_FIRST;
      selectedRepos = this.selectPriorityRepos(availableRepos);
    } else {
      // Default to scatter-gather
      strategy = SearchStrategy.SCATTER_GATHER;
      selectedRepos = availableRepos.slice(0, this.config.maxConcurrentRepos);
    }
    
    const expectedLatency = this.calculateExpectedLatency(selectedRepos);
    const confidenceScore = this.calculateConfidenceScore(selectedRepos, queryFeatures);
    
    return {
      repositories: selectedRepos,
      strategy,
      expectedLatency,
      confidenceScore
    };
  }
  
  /**
   * Analyze query features for routing decisions
   */
  private analyzeQueryFeatures(context: SearchContext): {
    isSpecific: boolean;
    isComplex: boolean;
    hasLanguageHint: boolean;
    hasFileTypeHint: boolean;
    expectedRelevantRepos: string[];
  } {
    const query = context.query.toLowerCase();
    
    // Check for specific patterns
    const isSpecific = /[A-Z][a-z]+[A-Z]|[a-z]+_[a-z]+|\.[a-z]+$/.test(context.query);
    const isComplex = context.query.length > 50 || /\s+(AND|OR|NOT)\s+/.test(context.query);
    const hasLanguageHint = /\.(ts|js|py|rs|go|java|cpp|c|rb|php)$/.test(query);
    const hasFileTypeHint = /\.(json|yaml|yml|md|txt|config)$/.test(query);
    
    // Find repositories that might be relevant
    const expectedRelevantRepos: string[] = [];
    for (const [repoId, repo] of this.repositories.entries()) {
      if (hasLanguageHint) {
        const langExtension = query.match(/\.([a-z]+)$/)?.[1];
        if (langExtension && repo.metadata.languages.includes(langExtension)) {
          expectedRelevantRepos.push(repoId);
        }
      }
      
      if (repo.metadata.tags.some(tag => query.includes(tag.toLowerCase()))) {
        expectedRelevantRepos.push(repoId);
      }
    }
    
    return {
      isSpecific,
      isComplex,
      hasLanguageHint,
      hasFileTypeHint,
      expectedRelevantRepos
    };
  }
  
  /**
   * Select repositories based on locality awareness
   */
  private selectLocalityAwareRepos(
    availableRepos: FederatedRepository[], 
    queryFeatures: any
  ): FederatedRepository[] {
    const localRepos = availableRepos.filter(r => r.isLocal);
    const remoteRepos = availableRepos.filter(r => !r.isLocal);
    
    // Prefer local repositories, but include high-priority remote repos
    const selected = [
      ...localRepos.slice(0, Math.ceil(this.config.maxConcurrentRepos * 0.7)),
      ...remoteRepos
        .filter(r => r.priority <= 2) // High priority only
        .slice(0, Math.floor(this.config.maxConcurrentRepos * 0.3))
    ];
    
    return selected.slice(0, this.config.maxConcurrentRepos);
  }
  
  /**
   * Select repositories using adaptive routing
   */
  private selectAdaptiveRepos(
    availableRepos: FederatedRepository[],
    queryFeatures: any
  ): FederatedRepository[] {
    const scored = availableRepos.map(repo => {
      let score = repo.weight * repo.healthScore;
      
      // Boost score for relevant repositories
      if (queryFeatures.expectedRelevantRepos.includes(repo.id)) {
        score *= 1.5;
      }
      
      // Boost score for local repositories
      if (repo.isLocal) {
        score *= 1.2;
      }
      
      // Penalize high-latency repositories
      const health = this.repositoryHealth.get(repo.id);
      if (health && health.responseTime > 1000) {
        score *= 0.8;
      }
      
      return { repo, score };
    });
    
    // Sort by score and take top repositories
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, this.config.maxConcurrentRepos).map(s => s.repo);
  }
  
  /**
   * Select repositories based on priority
   */
  private selectPriorityRepos(availableRepos: FederatedRepository[]): FederatedRepository[] {
    return availableRepos
      .sort((a, b) => a.priority - b.priority)
      .slice(0, this.config.maxConcurrentRepos);
  }
  
  /**
   * Merge search results using configured strategy
   */
  private mergeSearchResults(hits: SearchHit[], strategy: SearchStrategy): SearchHit[] {
    switch (this.config.resultMergingStrategy) {
      case 'score_weighted':
        return this.mergeByScoreWeighted(hits);
      case 'priority_based':
        return this.mergeByPriority(hits);
      case 'round_robin':
        return this.mergeRoundRobin(hits);
      default:
        return hits.sort((a, b) => (b.score || 0) - (a.score || 0));
    }
  }
  
  /**
   * Merge results using score weighting
   */
  private mergeByScoreWeighted(hits: SearchHit[]): SearchHit[] {
    // Normalize scores across repositories if enabled
    if (this.config.scoreNormalization) {
      hits = this.normalizeScores(hits);
    }
    
    // Sort by adjusted score
    return hits.sort((a, b) => {
      const aScore = (a.score || 0) * (a.repository?.weight || 1);
      const bScore = (b.score || 0) * (b.repository?.weight || 1);
      return bScore - aScore;
    });
  }
  
  /**
   * Merge results by repository priority
   */
  private mergeByPriority(hits: SearchHit[]): SearchHit[] {
    return hits.sort((a, b) => {
      const aPriority = a.repository?.priority || 999;
      const bPriority = b.repository?.priority || 999;
      
      if (aPriority !== bPriority) {
        return aPriority - bPriority;
      }
      
      return (b.score || 0) - (a.score || 0);
    });
  }
  
  /**
   * Merge results using round-robin from each repository
   */
  private mergeRoundRobin(hits: SearchHit[]): SearchHit[] {
    const repoGroups = new Map<string, SearchHit[]>();
    
    // Group hits by repository
    for (const hit of hits) {
      const repoId = hit.repository?.id || 'unknown';
      if (!repoGroups.has(repoId)) {
        repoGroups.set(repoId, []);
      }
      repoGroups.get(repoId)!.push(hit);
    }
    
    // Sort each group by score
    for (const group of repoGroups.values()) {
      group.sort((a, b) => (b.score || 0) - (a.score || 0));
    }
    
    // Round-robin merge
    const merged: SearchHit[] = [];
    const maxLength = Math.max(...Array.from(repoGroups.values()).map(g => g.length));
    
    for (let i = 0; i < maxLength; i++) {
      for (const group of repoGroups.values()) {
        if (i < group.length) {
          merged.push(group[i]);
        }
      }
    }
    
    return merged;
  }
  
  /**
   * Normalize scores across repositories
   */
  private normalizeScores(hits: SearchHit[]): SearchHit[] {
    const repoGroups = new Map<string, SearchHit[]>();
    
    // Group by repository
    for (const hit of hits) {
      const repoId = hit.repository?.id || 'unknown';
      if (!repoGroups.has(repoId)) {
        repoGroups.set(repoId, []);
      }
      repoGroups.get(repoId)!.push(hit);
    }
    
    // Normalize scores within each repository
    for (const [repoId, group] of repoGroups.entries()) {
      const scores = group.map(h => h.score || 0);
      const minScore = Math.min(...scores);
      const maxScore = Math.max(...scores);
      const scoreRange = maxScore - minScore;
      
      if (scoreRange > 0) {
        for (const hit of group) {
          const originalScore = hit.score || 0;
          hit.score = (originalScore - minScore) / scoreRange;
        }
      }
    }
    
    return hits;
  }
  
  /**
   * Deduplicate results across repositories
   */
  private deduplicateResults(hits: SearchHit[]): SearchHit[] {
    const seen = new Set<string>();
    const deduplicated: SearchHit[] = [];
    
    for (const hit of hits) {
      // Create deduplication key based on file path and content
      const dedupeKey = this.createDeduplicationKey(hit);
      
      if (!seen.has(dedupeKey)) {
        seen.add(dedupeKey);
        deduplicated.push(hit);
      } else {
        // Keep track of cross-repo hits
        this.searchStats.crossRepoHits++;
      }
    }
    
    return deduplicated;
  }
  
  /**
   * Create deduplication key for search hit
   */
  private createDeduplicationKey(hit: SearchHit): string {
    const components = [
      hit.file,
      hit.line?.toString() || '',
      hit.col?.toString() || '',
      (hit.snippet || '').substring(0, 100) // First 100 chars of snippet
    ];
    
    return createHash('md5').update(components.join('|')).digest('hex');
  }
  
  /**
   * Calculate expected latency for repository set
   */
  private calculateExpectedLatency(repos: FederatedRepository[]): number {
    if (repos.length === 0) return 0;
    
    const latencies = repos.map(repo => {
      const health = this.repositoryHealth.get(repo.id);
      return health?.responseTime || 100; // Default 100ms
    });
    
    // For parallel execution, expected latency is roughly the 95th percentile
    latencies.sort((a, b) => a - b);
    const p95Index = Math.floor(latencies.length * 0.95);
    return latencies[p95Index] || latencies[latencies.length - 1];
  }
  
  /**
   * Calculate confidence score for search route
   */
  private calculateConfidenceScore(repos: FederatedRepository[], queryFeatures: any): number {
    if (repos.length === 0) return 0;
    
    let score = 0;
    
    for (const repo of repos) {
      const health = this.repositoryHealth.get(repo.id);
      const healthScore = health?.successRate || 0.5;
      
      let repoScore = repo.weight * healthScore;
      
      // Boost for relevant repositories
      if (queryFeatures.expectedRelevantRepos.includes(repo.id)) {
        repoScore *= 1.3;
      }
      
      score += repoScore;
    }
    
    return Math.min(score / repos.length, 1.0);
  }
  
  /**
   * Perform health check on repository
   */
  private async performHealthCheck(repoId: string): Promise<void> {
    const span = LensTracer.createChildSpan('repo_health_check');
    
    try {
      const repo = this.repositories.get(repoId);
      if (!repo) return;
      
      const startTime = performance.now();
      let isHealthy = true;
      let error: string | undefined;
      
      if (repo.isLocal) {
        // Local repository health check
        isHealthy = await this.checkLocalRepositoryHealth(repo);
      } else {
        // Remote repository health check
        isHealthy = await this.checkRemoteRepositoryHealth(repo);
      }
      
      const responseTime = performance.now() - startTime;
      
      // Update health record
      const health = this.repositoryHealth.get(repoId);
      if (health) {
        health.isHealthy = isHealthy;
        health.responseTime = responseTime;
        health.lastHealthCheck = Date.now();
        
        if (!isHealthy) {
          health.errorCount++;
          health.issues.push(error || 'Health check failed');
        }
        
        // Update success rate (exponential moving average)
        const alpha = 0.1;
        health.successRate = (1 - alpha) * health.successRate + alpha * (isHealthy ? 1 : 0);
      }
      
      // Update repository health score
      repo.healthScore = health?.successRate || 0.5;
      
      span.setAttributes({
        success: true,
        repo_id: repoId,
        is_healthy: isHealthy,
        response_time: responseTime
      });
      
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      span.recordException(err as Error);
      span.setAttributes({ success: false, error: errorMsg });
    } finally {
      span.end();
    }
  }
  
  /**
   * Check local repository health
   */
  private async checkLocalRepositoryHealth(repo: FederatedRepository): Promise<boolean> {
    try {
      // Simple health check - try to access repository metadata
      return repo.isActive && repo.metadata.fileCount > 0;
    } catch {
      return false;
    }
  }
  
  /**
   * Check remote repository health
   */
  private async checkRemoteRepositoryHealth(repo: FederatedRepository): Promise<boolean> {
    try {
      const healthUrl = `${repo.searchEndpoint}/health`;
      const response = await fetch(healthUrl, {
        method: 'GET',
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });
      
      return response.ok;
    } catch {
      return false;
    }
  }
  
  /**
   * Check if repository is healthy
   */
  private isRepositoryHealthy(repoId: string): boolean {
    const health = this.repositoryHealth.get(repoId);
    if (!health) return false;
    
    // Consider healthy if success rate > 80% and recent response time < 5s
    return health.isHealthy && 
           health.successRate > 0.8 && 
           health.responseTime < 5000 &&
           (Date.now() - health.lastHealthCheck) < this.config.healthCheckInterval * 2;
  }
  
  /**
   * Update repository health based on search results
   */
  private updateRepositoryHealth(repoId: string, success: boolean, error?: string): void {
    const health = this.repositoryHealth.get(repoId);
    if (!health) return;
    
    // Update success rate
    const alpha = 0.1;
    health.successRate = (1 - alpha) * health.successRate + alpha * (success ? 1 : 0);
    
    if (!success) {
      health.errorCount++;
      if (error) {
        health.issues.push(error);
        
        // Keep only recent issues
        if (health.issues.length > 10) {
          health.issues = health.issues.slice(-5);
        }
      }
    }
    
    // Update repository health score
    const repo = this.repositories.get(repoId);
    if (repo) {
      repo.healthScore = health.successRate;
    }
  }
  
  /**
   * Update routing table based on repository states
   */
  private updateRoutingTable(): void {
    this.routingTable.clear();
    
    // This would implement more sophisticated routing table updates
    // For now, just clear to force recalculation
    console.log('🔄 Updated federation routing table');
  }
  
  /**
   * Update search statistics
   */
  private updateSearchStats(latency: number, wasFederated: boolean): void {
    // Update average latency (exponential moving average)
    const alpha = 0.1;
    this.searchStats.avgLatency = (1 - alpha) * this.searchStats.avgLatency + alpha * latency;
    
    if (wasFederated) {
      this.searchStats.federatedSearches++;
    }
    
    // Calculate success rate
    const totalComplete = this.searchStats.totalSearches;
    this.searchStats.successRate = totalComplete > 0 ? 
      (totalComplete - 0) / totalComplete : 0; // Simplified - would track failures
  }
  
  /**
   * Generate repository ID
   */
  private generateRepositoryId(name: string, url: string): string {
    const data = `${name}:${url}:${Date.now()}`;
    return createHash('sha256').update(data).digest('hex').substring(0, 16);
  }
  
  /**
   * Generate federated cache key
   */
  private generateFederatedCacheKey(context: SearchContext): string {
    const key = JSON.stringify({
      query: context.query,
      mode: context.mode,
      k: context.k,
      fuzzy_distance: context.fuzzy_distance
    });
    return `fed_search_${createHash('md5').update(key).digest('hex')}`;
  }
  
  /**
   * Add timeout to promise
   */
  private withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
    return Promise.race([
      promise,
      new Promise<T>((_, reject) => 
        setTimeout(() => reject(new Error('Operation timed out')), timeoutMs)
      )
    ]);
  }
  
  /**
   * Start maintenance timers
   */
  private startMaintenanceTimers(): void {
    // Health checks
    this.healthCheckTimer = setInterval(() => {
      for (const repoId of this.repositories.keys()) {
        this.performHealthCheck(repoId).catch(error => 
          console.warn(`Health check failed for repository ${repoId}:`, error)
        );
      }
    }, this.config.healthCheckInterval);
    
    // Routing optimization
    this.routingOptimizationTimer = setInterval(() => {
      this.updateRoutingTable();
    }, 300000); // 5 minutes
  }
  
  /**
   * Get federation statistics
   */
  getStats(): {
    registeredRepos: number;
    activeRepos: number;
    healthyRepos: number;
    searchStats: typeof this.searchStats;
    avgResponseTime: number;
  } {
    const activeRepos = Array.from(this.repositories.values()).filter(r => r.isActive).length;
    const healthyRepos = Array.from(this.repositories.keys()).filter(r => this.isRepositoryHealthy(r)).length;
    
    const responseTimes = Array.from(this.repositoryHealth.values()).map(h => h.responseTime);
    const avgResponseTime = responseTimes.length > 0 ? 
      responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length : 0;
    
    return {
      registeredRepos: this.repositories.size,
      activeRepos,
      healthyRepos,
      searchStats: this.searchStats,
      avgResponseTime
    };
  }
  
  /**
   * Shutdown federation manager
   */
  shutdown(): void {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
    }
    
    if (this.routingOptimizationTimer) {
      clearInterval(this.routingOptimizationTimer);
    }
    
    this.repositories.clear();
    this.repositoryHealth.clear();
    this.routingTable.clear();
    
    console.log('🛑 Federation Manager shutdown complete');
  }
}

// Global instance
export const globalFederationManager = FederationManager.getInstance();