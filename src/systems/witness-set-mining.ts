/**
 * Task-Level Correctness with Witness Set Mining
 * 
 * Mathematical target: maximize Success@k where result set S_k is success iff it covers witness set W: S_k ∩ W ≠ ∅
 * Minimizes |W| via greedy set cover over def-use and build edges
 */

import type { SearchHit, SearchContext } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';
import { readFile, writeFile, readdir } from 'fs/promises';
import { join, dirname } from 'path';
import { existsSync } from 'fs';

export interface WitnessSet {
  readonly id: string;
  readonly query: string;
  readonly files: readonly string[];
  readonly minimalHittingSet: readonly string[];
  readonly confidence: number;
  readonly source: 'ci_failure' | 'bug_fix_commit' | 'manual_label';
  readonly metadata: {
    readonly commit_sha?: string;
    readonly test_failure?: string;
    readonly build_error?: string;
    readonly created_at: Date;
  };
}

export interface WitnessSetMiningConfig {
  readonly maxWitnessSize: number;
  readonly minConfidence: number;
  readonly greedySetCoverThreshold: number;
  readonly ciFailureWindowHours: number;
  readonly bugFixPatternMatchers: readonly RegExp[];
}

export interface TaskSuccess {
  readonly query: string;
  readonly resultSet: readonly string[];
  readonly witnessSet: readonly string[];
  readonly success: boolean;
  readonly coverage: number;
  readonly timestamp: Date;
}

export interface SuccessAtK {
  readonly k: number;
  readonly successRate: number;
  readonly totalQueries: number;
  readonly successfulQueries: number;
  readonly averageCoverage: number;
}

const DEFAULT_CONFIG: WitnessSetMiningConfig = {
  maxWitnessSize: 10,
  minConfidence: 0.7,
  greedySetCoverThreshold: 0.9,
  ciFailureWindowHours: 24,
  bugFixPatternMatchers: [
    /fix\s+bug/i,
    /resolve\s+issue/i,
    /patch\s+for/i,
    /\bfix\b.*\berror\b/i,
    /\bfixes?\b\s+#\d+/i,
  ],
};

export class WitnessSetMiner {
  private witnessSetCache = new Map<string, WitnessSet>();
  private successHistory: TaskSuccess[] = [];
  private config: WitnessSetMiningConfig;

  constructor(
    private ciLogsPath: string,
    private gitRepoPath: string,
    config: Partial<WitnessSetMiningConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Mine witness sets from CI failures and bug-fix commits
   */
  async mineWitnessSets(): Promise<WitnessSet[]> {
    const span = LensTracer.createChildSpan('mine_witness_sets');
    
    try {
      const [ciWitnessSets, bugFixWitnessSets] = await Promise.all([
        this.mineCIFailureWitnessSets(),
        this.mineBugFixCommitWitnessSets(),
      ]);

      const allWitnessSets = [...ciWitnessSets, ...bugFixWitnessSets];
      
      // Apply greedy set cover to minimize witness set sizes
      const optimizedWitnessSets = await Promise.all(
        allWitnessSets.map(ws => this.applyGreedySetCover(ws))
      );

      // Cache witness sets for fast lookup
      for (const ws of optimizedWitnessSets) {
        this.witnessSetCache.set(ws.id, ws);
      }

      span.setAttributes({
        success: true,
        ci_witness_sets: ciWitnessSets.length,
        bug_fix_witness_sets: bugFixWitnessSets.length,
        total_witness_sets: optimizedWitnessSets.length,
      });

      return optimizedWitnessSets;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Evaluate Success@k for a given result set against known witness sets
   */
  evaluateSuccessAtK(resultSet: readonly string[], witnessSet: readonly string[]): boolean {
    // Success iff result set covers witness set: S_k ∩ W ≠ ∅
    const resultSetSet = new Set(resultSet);
    return witnessSet.some(file => resultSetSet.has(file));
  }

  /**
   * Record a task success/failure for metrics
   */
  recordTaskResult(
    query: string,
    resultSet: readonly string[],
    witnessSet: readonly string[]
  ): TaskSuccess {
    const success = this.evaluateSuccessAtK(resultSet, witnessSet);
    const coverage = this.calculateCoverage(resultSet, witnessSet);
    
    const result: TaskSuccess = {
      query,
      resultSet,
      witnessSet,
      success,
      coverage,
      timestamp: new Date(),
    };

    this.successHistory.push(result);
    
    // Keep only last 10000 results to prevent memory bloat
    if (this.successHistory.length > 10000) {
      this.successHistory = this.successHistory.slice(-5000);
    }

    return result;
  }

  /**
   * Calculate Success@K metrics across all recorded results
   */
  calculateSuccessAtK(k: number): SuccessAtK {
    const relevantResults = this.successHistory.filter(
      result => result.resultSet.length <= k
    );

    const successfulQueries = relevantResults.filter(r => r.success).length;
    const totalQueries = relevantResults.length;
    const successRate = totalQueries > 0 ? successfulQueries / totalQueries : 0;
    const averageCoverage = relevantResults.length > 0
      ? relevantResults.reduce((sum, r) => sum + r.coverage, 0) / relevantResults.length
      : 0;

    return {
      k,
      successRate,
      totalQueries,
      successfulQueries,
      averageCoverage,
    };
  }

  /**
   * Get witness set features for Stage-C ranking
   */
  getWitnessSetFeatures(query: string, hits: readonly SearchHit[]): number[] {
    const witnessSet = this.findBestWitnessSet(query);
    if (!witnessSet) {
      return [0, 0, 0]; // [coverage, confidence, size]
    }

    const hitFiles = hits.map(h => h.file);
    const coverage = this.calculateCoverage(hitFiles, witnessSet.files);
    
    return [
      coverage,
      witnessSet.confidence,
      1.0 / witnessSet.minimalHittingSet.length, // Smaller witness sets are better
    ];
  }

  /**
   * Check if current SLA-Success@10 is flat or increasing
   */
  checkSLASuccessAt10(): { isFlat: boolean; trend: number; current: number } {
    const recentWindow = 1000; // Last 1000 queries
    const recent = this.successHistory.slice(-recentWindow);
    const historical = this.successHistory.slice(-recentWindow * 2, -recentWindow);

    const recentSuccess = this.calculateSuccessAtK(10);
    const historicalSuccess = historical.length > 0 
      ? this.calculateSuccessAtKForResults(historical, 10)
      : recentSuccess;

    const trend = recentSuccess.successRate - historicalSuccess.successRate;
    const isFlat = Math.abs(trend) < 0.05; // Within 5% is considered flat

    return {
      isFlat: isFlat || trend >= 0,
      trend,
      current: recentSuccess.successRate,
    };
  }

  /**
   * Mine witness sets from CI failure logs
   */
  private async mineCIFailureWitnessSets(): Promise<WitnessSet[]> {
    const witnessSets: WitnessSet[] = [];
    
    if (!existsSync(this.ciLogsPath)) {
      return witnessSets;
    }

    try {
      const logFiles = await readdir(this.ciLogsPath);
      const recentLogs = logFiles
        .filter(f => f.endsWith('.log'))
        .slice(-50); // Process last 50 CI logs

      for (const logFile of recentLogs) {
        const logPath = join(this.ciLogsPath, logFile);
        const content = await readFile(logPath, 'utf8');
        
        const failurePatterns = this.extractFailurePatterns(content);
        for (const pattern of failurePatterns) {
          const files = await this.findRelatedFiles(pattern.query, pattern.context);
          
          if (files.length > 0 && files.length <= this.config.maxWitnessSize) {
            witnessSets.push({
              id: `ci_${logFile}_${pattern.query.replace(/\W/g, '_')}`,
              query: pattern.query,
              files,
              minimalHittingSet: files, // Will be optimized later
              confidence: this.calculateConfidenceFromCI(pattern),
              source: 'ci_failure',
              metadata: {
                test_failure: pattern.testName,
                build_error: pattern.error,
                created_at: new Date(),
              },
            });
          }
        }
      }
    } catch (error) {
      console.warn(`Failed to mine CI witness sets: ${error}`);
    }

    return witnessSets;
  }

  /**
   * Mine witness sets from bug-fix commits
   */
  private async mineBugFixCommitWitnessSets(): Promise<WitnessSet[]> {
    const witnessSets: WitnessSet[] = [];
    
    try {
      const commitHistory = await this.getRecentCommits(100);
      const bugFixCommits = commitHistory.filter(commit =>
        this.config.bugFixPatternMatchers.some(pattern =>
          pattern.test(commit.message)
        )
      );

      for (const commit of bugFixCommits) {
        const changedFiles = await this.getChangedFiles(commit.sha);
        const query = this.extractQueryFromCommitMessage(commit.message);
        
        if (query && changedFiles.length > 0 && changedFiles.length <= this.config.maxWitnessSize) {
          witnessSets.push({
            id: `bugfix_${commit.sha}`,
            query,
            files: changedFiles,
            minimalHittingSet: changedFiles, // Will be optimized later
            confidence: this.calculateConfidenceFromBugFix(commit),
            source: 'bug_fix_commit',
            metadata: {
              commit_sha: commit.sha,
              created_at: new Date(commit.timestamp),
            },
          });
        }
      }
    } catch (error) {
      console.warn(`Failed to mine bug-fix witness sets: ${error}`);
    }

    return witnessSets;
  }

  /**
   * Apply greedy set cover to minimize witness set size
   */
  private async applyGreedySetCover(witnessSet: WitnessSet): Promise<WitnessSet> {
    // Build def-use and build dependency graph
    const dependencyGraph = await this.buildDependencyGraph(witnessSet.files);
    
    // Apply greedy set cover algorithm
    const minimalSet = this.greedySetCover(witnessSet.files, dependencyGraph);
    
    return {
      ...witnessSet,
      minimalHittingSet: minimalSet,
    };
  }

  /**
   * Greedy set cover algorithm to minimize witness set
   */
  private greedySetCover(universe: readonly string[], dependencies: Map<string, Set<string>>): string[] {
    const uncovered = new Set(universe);
    const cover: string[] = [];
    
    while (uncovered.size > 0) {
      let bestFile: string | null = null;
      let bestCoverage = 0;
      
      for (const file of universe) {
        if (cover.includes(file)) continue;
        
        // Count how many uncovered elements this file covers (including transitive dependencies)
        const covered = this.getTransitiveDependencies(file, dependencies);
        const coverageCount = Array.from(covered).filter(f => uncovered.has(f)).length;
        
        if (coverageCount > bestCoverage) {
          bestCoverage = coverageCount;
          bestFile = file;
        }
      }
      
      if (bestFile) {
        cover.push(bestFile);
        const covered = this.getTransitiveDependencies(bestFile, dependencies);
        for (const file of covered) {
          uncovered.delete(file);
        }
      } else {
        // No more files can cover remaining elements
        break;
      }
    }
    
    return cover;
  }

  /**
   * Build dependency graph from def-use and build edges
   */
  private async buildDependencyGraph(files: readonly string[]): Promise<Map<string, Set<string>>> {
    const dependencies = new Map<string, Set<string>>();
    
    for (const file of files) {
      const deps = new Set<string>();
      
      try {
        const content = await readFile(join(this.gitRepoPath, file), 'utf8');
        
        // Extract import/require dependencies
        const importMatches = content.matchAll(/(?:import|require)\s*\(\s*['"`]([^'"`]+)['"`]\s*\)/g);
        for (const match of importMatches) {
          const importPath = this.resolveImportPath(file, match[1]);
          if (importPath && files.includes(importPath)) {
            deps.add(importPath);
          }
        }
        
        // Extract ES6 import dependencies
        const es6ImportMatches = content.matchAll(/import\s+.*\s+from\s+['"`]([^'"`]+)['"`]/g);
        for (const match of es6ImportMatches) {
          const importPath = this.resolveImportPath(file, match[1]);
          if (importPath && files.includes(importPath)) {
            deps.add(importPath);
          }
        }
        
      } catch (error) {
        // File might not exist or be readable, skip
      }
      
      dependencies.set(file, deps);
    }
    
    return dependencies;
  }

  /**
   * Get transitive dependencies for a file
   */
  private getTransitiveDependencies(file: string, dependencies: Map<string, Set<string>>): Set<string> {
    const visited = new Set<string>();
    const result = new Set<string>();
    
    const dfs = (currentFile: string) => {
      if (visited.has(currentFile)) return;
      visited.add(currentFile);
      result.add(currentFile);
      
      const deps = dependencies.get(currentFile) || new Set();
      for (const dep of deps) {
        dfs(dep);
      }
    };
    
    dfs(file);
    return result;
  }

  /**
   * Calculate coverage between result set and witness set
   */
  private calculateCoverage(resultSet: readonly string[], witnessSet: readonly string[]): number {
    if (witnessSet.length === 0) return 1.0;
    
    const resultSetSet = new Set(resultSet);
    const coveredFiles = witnessSet.filter(file => resultSetSet.has(file)).length;
    
    return coveredFiles / witnessSet.length;
  }

  /**
   * Find the best witness set for a query
   */
  private findBestWitnessSet(query: string): WitnessSet | null {
    let bestMatch: WitnessSet | null = null;
    let bestScore = 0;
    
    for (const witnessSet of this.witnessSetCache.values()) {
      const similarity = this.calculateQuerySimilarity(query, witnessSet.query);
      const score = similarity * witnessSet.confidence;
      
      if (score > bestScore && score >= this.config.minConfidence) {
        bestScore = score;
        bestMatch = witnessSet;
      }
    }
    
    return bestMatch;
  }

  /**
   * Calculate similarity between two queries
   */
  private calculateQuerySimilarity(query1: string, query2: string): number {
    const words1 = query1.toLowerCase().split(/\s+/);
    const words2 = query2.toLowerCase().split(/\s+/);
    
    const intersection = words1.filter(word => words2.includes(word));
    const union = [...new Set([...words1, ...words2])];
    
    return union.length > 0 ? intersection.length / union.length : 0;
  }

  /**
   * Calculate Success@K for a specific set of results
   */
  private calculateSuccessAtKForResults(results: TaskSuccess[], k: number): SuccessAtK {
    const relevantResults = results.filter(result => result.resultSet.length <= k);
    const successfulQueries = relevantResults.filter(r => r.success).length;
    const totalQueries = relevantResults.length;
    
    return {
      k,
      successRate: totalQueries > 0 ? successfulQueries / totalQueries : 0,
      totalQueries,
      successfulQueries,
      averageCoverage: relevantResults.length > 0
        ? relevantResults.reduce((sum, r) => sum + r.coverage, 0) / relevantResults.length
        : 0,
    };
  }

  // Placeholder implementations for git operations and file analysis
  private extractFailurePatterns(content: string): { query: string; context: string; testName?: string; error?: string }[] {
    const patterns: { query: string; context: string; testName?: string; error?: string }[] = [];
    
    // Extract test failures
    const testFailureRegex = /FAIL:\s+(.+?)\s+\((.+?)\)/g;
    let match;
    while ((match = testFailureRegex.exec(content)) !== null) {
      patterns.push({
        query: match[1].replace(/[_-]/g, ' ').trim(),
        context: match[2],
        testName: match[1],
      });
    }
    
    // Extract build errors
    const buildErrorRegex = /ERROR:\s+(.+?)(?:\n|$)/g;
    while ((match = buildErrorRegex.exec(content)) !== null) {
      patterns.push({
        query: match[1].replace(/[^\w\s]/g, ' ').trim(),
        context: 'build_error',
        error: match[1],
      });
    }
    
    return patterns;
  }

  private async findRelatedFiles(query: string, context: string): Promise<string[]> {
    // Simple heuristic: look for files containing query terms
    const queryTerms = query.toLowerCase().split(/\s+/);
    const files: string[] = [];
    
    try {
      // This would normally use git or filesystem traversal
      // For now, return empty array as placeholder
      return files;
    } catch {
      return [];
    }
  }

  private calculateConfidenceFromCI(pattern: { query: string; context: string }): number {
    // Higher confidence for test failures, lower for build errors
    if (pattern.context.includes('test')) return 0.9;
    if (pattern.context.includes('build')) return 0.7;
    return 0.5;
  }

  private async getRecentCommits(count: number): Promise<{ sha: string; message: string; timestamp: string }[]> {
    // Placeholder for git log parsing
    return [];
  }

  private calculateConfidenceFromBugFix(commit: { message: string }): number {
    // Higher confidence for explicit bug fixes
    if (/\bfix\b.*\bbug\b/i.test(commit.message)) return 0.9;
    if (/\bfixes?\b\s+#\d+/i.test(commit.message)) return 0.8;
    return 0.6;
  }

  private extractQueryFromCommitMessage(message: string): string | null {
    // Extract meaningful query from commit message
    const cleaned = message
      .replace(/^(fix|resolve|patch|update):\s*/i, '')
      .replace(/#\d+/g, '')
      .replace(/\([^)]+\)$/g, '')
      .trim();
    
    return cleaned.length > 3 ? cleaned : null;
  }

  private async getChangedFiles(commitSha: string): Promise<string[]> {
    // Placeholder for git diff parsing
    return [];
  }

  private resolveImportPath(fromFile: string, importPath: string): string | null {
    // Simplified import resolution
    if (importPath.startsWith('.')) {
      const dir = dirname(fromFile);
      return join(dir, importPath).replace(/\\/g, '/');
    }
    return null;
  }
}