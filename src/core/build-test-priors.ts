/**
 * Build/Test-Aware Priors - Evergreen Optimization System #2
 * 
 * Ingest CI/build graphs (Bazel/Gradle/Cargo) and recent test failures
 * Compute bounded prior B(file) from: transitive deps to failing targets, change churn, codeowner proximity
 * Use B as tie-break/path prior in Stage-A and capped feature in Stage-C (|Δlog-odds|≤0.3)
 * 24-48h decay half-life to prevent chasing red tests
 * 
 * Gate: ΔnDCG@10 ≥ +0.5pp on failure-adjacent queries, SLA-Recall@50 ≥ 0, Core@10 drift ≤±5pp topic-normalized
 * Track why-mix KL to avoid crowding out semantic wins
 */

import type { SearchContext, Candidate } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';

export interface BuildTarget {
  id: string;
  name: string;
  type: 'binary' | 'library' | 'test' | 'data';
  sources: string[]; // source file paths
  deps: string[]; // dependency target ids
  test_size?: 'small' | 'medium' | 'large';
  tags: string[];
  package_path: string;
}

export interface TestFailure {
  target_id: string;
  test_name: string;
  failure_time: Date;
  failure_type: 'build' | 'test' | 'timeout';
  error_message?: string;
  stack_trace?: string;
  affected_files: string[];
  build_log?: string;
}

export interface ChangeEvent {
  commit_sha: string;
  timestamp: Date;
  author: string;
  files_changed: string[];
  lines_added: number;
  lines_deleted: number;
  change_type: 'feature' | 'bugfix' | 'refactor' | 'test' | 'docs';
}

export interface CodeOwner {
  pattern: string; // glob pattern
  owners: string[];
  confidence: number; // 0-1, based on recent activity
}

export interface FilePrior {
  file_path: string;
  prior_score: number; // B(file)
  components: {
    failure_proximity: number; // distance to failing tests
    change_churn: number; // recent change frequency
    owner_proximity: number; // distance to code owners
  };
  last_updated: Date;
  decay_factor: number; // based on time since last update
}

/**
 * Build graph parser for different build systems
 */
export abstract class BuildGraphParser {
  abstract parseBuildFile(content: string, filePath: string): BuildTarget[];
  abstract detectBuildSystem(filePath: string): boolean;
  
  protected extractDependencies(depString: string): string[] {
    // Common patterns across build systems
    const deps: string[] = [];
    const patterns = [
      /"([^"]+)"/g, // quoted deps
      /'([^']+)'/g, // single quoted
      /\b[\w\-_./:]+\b/g, // unquoted identifiers
    ];
    
    for (const pattern of patterns) {
      const matches = depString.matchAll(pattern);
      for (const match of matches) {
        const dep = match[1] || match[0];
        if (dep && dep.length > 1 && !dep.startsWith('//')) {
          deps.push(dep);
        }
      }
    }
    
    return deps;
  }
}

/**
 * Bazel BUILD file parser
 */
export class BazelParser extends BuildGraphParser {
  detectBuildSystem(filePath: string): boolean {
    return filePath.endsWith('BUILD') || 
           filePath.endsWith('BUILD.bazel') ||
           filePath.includes('BUILD.');
  }

  parseBuildFile(content: string, filePath: string): BuildTarget[] {
    const targets: BuildTarget[] = [];
    const packagePath = filePath.substring(0, filePath.lastIndexOf('/'));
    
    // Parse Bazel rules (simplified regex-based approach)
    const rulePattern = /(\w+)\s*\(\s*name\s*=\s*["']([^"']+)["']/g;
    const matches = content.matchAll(rulePattern);
    
    for (const match of matches) {
      const ruleType = match[1];
      const targetName = match[2];
      
      // Extract the full rule definition
      const ruleStart = match.index!;
      const ruleContent = this.extractRuleContent(content, ruleStart);
      
      const target: BuildTarget = {
        id: `${packagePath}:${targetName}`,
        name: targetName,
        type: this.mapBazelRuleToType(ruleType),
        sources: this.extractSources(ruleContent),
        deps: this.extractDependencies(this.extractAttribute(ruleContent, 'deps')),
        tags: this.extractTags(ruleContent),
        package_path: packagePath,
      };
      
      if (ruleType.includes('test')) {
        target.test_size = this.extractTestSize(ruleContent);
      }
      
      targets.push(target);
    }
    
    return targets;
  }

  private extractRuleContent(content: string, startIndex: number): string {
    let depth = 0;
    let i = startIndex;
    
    // Find opening parenthesis
    while (i < content.length && content[i] !== '(') i++;
    if (i >= content.length) return '';
    
    const start = i;
    
    // Find matching closing parenthesis
    for (; i < content.length; i++) {
      if (content[i] === '(') depth++;
      else if (content[i] === ')') depth--;
      if (depth === 0) break;
    }
    
    return content.substring(start, i + 1);
  }

  private extractAttribute(ruleContent: string, attrName: string): string {
    const pattern = new RegExp(`${attrName}\\s*=\\s*\\[(.*?)\\]`, 's');
    const match = ruleContent.match(pattern);
    return match ? match[1] : '';
  }

  private extractSources(ruleContent: string): string[] {
    const srcs = this.extractAttribute(ruleContent, 'srcs');
    return this.extractDependencies(srcs);
  }

  private extractTags(ruleContent: string): string[] {
    const tags = this.extractAttribute(ruleContent, 'tags');
    return this.extractDependencies(tags);
  }

  private extractTestSize(ruleContent: string): 'small' | 'medium' | 'large' {
    const sizeMatch = ruleContent.match(/size\s*=\s*["']([^"']+)["']/);
    const size = sizeMatch ? sizeMatch[1] : 'small';
    return ['small', 'medium', 'large'].includes(size) ? size as any : 'small';
  }

  private mapBazelRuleToType(ruleType: string): 'binary' | 'library' | 'test' | 'data' {
    if (ruleType.includes('test')) return 'test';
    if (ruleType.includes('binary')) return 'binary';
    if (ruleType.includes('library')) return 'library';
    return 'library';
  }
}

/**
 * Gradle build.gradle parser
 */
export class GradleParser extends BuildGraphParser {
  detectBuildSystem(filePath: string): boolean {
    return filePath.endsWith('build.gradle') ||
           filePath.endsWith('build.gradle.kts');
  }

  parseBuildFile(content: string, filePath: string): BuildTarget[] {
    const targets: BuildTarget[] = [];
    const packagePath = filePath.substring(0, filePath.lastIndexOf('/'));
    
    // Parse dependencies block
    const depsMatch = content.match(/dependencies\s*\{([^}]*)\}/s);
    const deps = depsMatch ? this.extractGradleDeps(depsMatch[1]) : [];
    
    // Create a synthetic target for the module
    const moduleName = packagePath.split('/').pop() || 'main';
    targets.push({
      id: `${packagePath}:${moduleName}`,
      name: moduleName,
      type: 'library',
      sources: [], // Would need to infer from source sets
      deps,
      tags: [],
      package_path: packagePath,
    });
    
    return targets;
  }

  private extractGradleDeps(depsContent: string): string[] {
    const deps: string[] = [];
    const patterns = [
      /(?:implementation|api|compileOnly|testImplementation)\s+["']([^"']+)["']/g,
      /(?:implementation|api|compileOnly|testImplementation)\s+project\s*\(\s*["']:([^"']+)["']\s*\)/g,
    ];
    
    for (const pattern of patterns) {
      const matches = depsContent.matchAll(pattern);
      for (const match of matches) {
        deps.push(match[1]);
      }
    }
    
    return deps;
  }
}

/**
 * Cargo.toml parser
 */
export class CargoParser extends BuildGraphParser {
  detectBuildSystem(filePath: string): boolean {
    return filePath.endsWith('Cargo.toml');
  }

  parseBuildFile(content: string, filePath: string): BuildTarget[] {
    const targets: BuildTarget[] = [];
    const packagePath = filePath.substring(0, filePath.lastIndexOf('/'));
    
    // Parse [dependencies] section
    const depsMatch = content.match(/\[dependencies\](.*?)(?=\[|\z)/s);
    const deps = depsMatch ? this.extractCargoDeps(depsMatch[1]) : [];
    
    // Parse package name
    const nameMatch = content.match(/name\s*=\s*["']([^"']+)["']/);
    const packageName = nameMatch ? nameMatch[1] : 'main';
    
    targets.push({
      id: `${packagePath}:${packageName}`,
      name: packageName,
      type: 'library',
      sources: [], // Would need to infer from src/
      deps,
      tags: [],
      package_path: packagePath,
    });
    
    return targets;
  }

  private extractCargoDeps(depsContent: string): string[] {
    const deps: string[] = [];
    const lines = depsContent.split('\n');
    
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;
      
      const match = trimmed.match(/^([a-zA-Z0-9_-]+)\s*=/);
      if (match) {
        deps.push(match[1]);
      }
    }
    
    return deps;
  }
}

/**
 * Build/Test aware priors system
 */
export class BuildTestPriors {
  private buildTargets: Map<string, BuildTarget> = new Map();
  private testFailures: Map<string, TestFailure[]> = new Map(); // target_id -> failures
  private changeEvents: ChangeEvent[] = [];
  private codeOwners: CodeOwner[] = [];
  private filePriors: Map<string, FilePrior> = new Map();
  private parsers: BuildGraphParser[] = [];
  
  // Configuration
  private readonly decayHalfLife = 36 * 60 * 60 * 1000; // 36 hours in ms (24-48h range)
  private readonly maxLogOddsDelta = 0.3; // |Δlog-odds|≤0.3 constraint
  private enabled = false;

  constructor() {
    this.parsers = [
      new BazelParser(),
      new GradleParser(), 
      new CargoParser(),
    ];
  }

  /**
   * Enable build/test priors system
   */
  enable(): void {
    this.enabled = true;
  }

  /**
   * Ingest build files from the repository
   */
  async ingestBuildGraph(files: { path: string; content: string }[]): Promise<void> {
    const span = LensTracer.createChildSpan('ingest_build_graph', {
      'files.count': files.length,
    });

    try {
      for (const file of files) {
        const parser = this.parsers.find(p => p.detectBuildSystem(file.path));
        if (parser) {
          const targets = parser.parseBuildFile(file.content, file.path);
          for (const target of targets) {
            this.buildTargets.set(target.id, target);
          }
        }
      }

      span.setAttributes({
        success: true,
        'targets.parsed': this.buildTargets.size,
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
   * Record a test failure
   */
  recordTestFailure(failure: TestFailure): void {
    const failures = this.testFailures.get(failure.target_id) || [];
    failures.push(failure);
    this.testFailures.set(failure.target_id, failures);
    
    // Update file priors for affected files
    this.updateFilePriorsForFailure(failure);
  }

  /**
   * Record a change event
   */
  recordChangeEvent(change: ChangeEvent): void {
    this.changeEvents.push(change);
    
    // Keep only recent changes (last 7 days)
    const cutoff = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
    this.changeEvents = this.changeEvents.filter(c => c.timestamp > cutoff);
    
    // Update file priors for changed files
    this.updateFilePriorsForChange(change);
  }

  /**
   * Update code ownership information
   */
  updateCodeOwners(owners: CodeOwner[]): void {
    this.codeOwners = owners;
  }

  /**
   * Get prior score B(file) for a file
   */
  getFilePrior(filePath: string): number {
    if (!this.enabled) return 0;

    const prior = this.filePriors.get(filePath);
    if (!prior) return 0;

    // Apply decay based on time since last update
    const now = Date.now();
    const timeDiff = now - prior.last_updated.getTime();
    const decayFactor = Math.exp(-timeDiff / this.decayHalfLife);
    
    return prior.prior_score * decayFactor;
  }

  /**
   * Apply file prior as tie-break in Stage-A candidates
   */
  applyStageATieBreak(candidates: Candidate[]): Candidate[] {
    if (!this.enabled || candidates.length === 0) return candidates;

    const span = LensTracer.createChildSpan('apply_stage_a_tiebreak', {
      'candidates.count': candidates.length,
    });

    try {
      // Group candidates by score for tie-breaking
      const scoreGroups = new Map<number, Candidate[]>();
      for (const candidate of candidates) {
        const score = Math.round(candidate.score * 1000) / 1000; // Round to 3 decimals
        const group = scoreGroups.get(score) || [];
        group.push(candidate);
        scoreGroups.set(score, group);
      }

      // Apply tie-breaking within score groups
      const result: Candidate[] = [];
      for (const [score, group] of scoreGroups) {
        if (group.length === 1) {
          result.push(...group);
        } else {
          // Sort by file prior within tie group
          group.sort((a, b) => {
            const priorA = this.getFilePrior(a.file_path);
            const priorB = this.getFilePrior(b.file_path);
            return priorB - priorA; // Higher prior first
          });
          result.push(...group);
        }
      }

      span.setAttributes({
        success: true,
        'score_groups.count': scoreGroups.size,
      });

      return result;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      return candidates;
    } finally {
      span.end();
    }
  }

  /**
   * Apply capped feature in Stage-C with |Δlog-odds|≤0.3 constraint
   */
  applyStageCFeature(candidates: Candidate[]): Candidate[] {
    if (!this.enabled || candidates.length === 0) return candidates;

    const span = LensTracer.createChildSpan('apply_stage_c_feature', {
      'candidates.count': candidates.length,
    });

    try {
      for (const candidate of candidates) {
        const prior = this.getFilePrior(candidate.file_path);
        if (prior > 0) {
          // Convert prior to log-odds adjustment, capped at ±0.3
          const logOddsAdjustment = Math.max(-this.maxLogOddsDelta, 
                                           Math.min(this.maxLogOddsDelta, 
                                                  Math.log(prior + 1) - Math.log(2)));
          
          // Convert back to score multiplier
          const scoreMultiplier = Math.exp(logOddsAdjustment);
          candidate.score *= scoreMultiplier;
          
          // Track that build prior was applied
          if (!candidate.match_reasons.includes('build_prior' as any)) {
            candidate.match_reasons.push('build_prior' as any);
          }
        }
      }

      span.setAttributes({ success: true });
      return candidates;
      
    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      return candidates;
    } finally {
      span.end();
    }
  }

  /**
   * Check if query is failure-adjacent (for gate metric)
   */
  isFailureAdjacentQuery(query: string, context: SearchContext): boolean {
    const queryLower = query.toLowerCase();
    
    // Check if query mentions test/build/error terms
    const failureKeywords = ['test', 'fail', 'error', 'exception', 'build', 'compile', 'timeout'];
    const hasFailureKeyword = failureKeywords.some(keyword => queryLower.includes(keyword));
    
    if (!hasFailureKeyword) return false;
    
    // Check if there are recent failures in the codebase
    const recentCutoff = new Date(Date.now() - 24 * 60 * 60 * 1000);
    const hasRecentFailures = Array.from(this.testFailures.values())
      .flat()
      .some(failure => failure.failure_time > recentCutoff);
    
    return hasRecentFailures;
  }

  // Private helper methods

  private updateFilePriorsForFailure(failure: TestFailure): void {
    const target = this.buildTargets.get(failure.target_id);
    if (!target) return;

    // Update priors for all transitive dependencies
    const affectedFiles = new Set<string>();
    this.collectTransitiveDependencies(target, affectedFiles);

    for (const filePath of affectedFiles) {
      const existing = this.filePriors.get(filePath) || this.createEmptyFilePrior(filePath);
      
      // Increase failure proximity component
      existing.components.failure_proximity = Math.min(1.0, 
        existing.components.failure_proximity + 0.3);
      
      this.updateFilePriorScore(existing);
      this.filePriors.set(filePath, existing);
    }
  }

  private updateFilePriorsForChange(change: ChangeEvent): void {
    for (const filePath of change.files_changed) {
      const existing = this.filePriors.get(filePath) || this.createEmptyFilePrior(filePath);
      
      // Update change churn component
      const changeWeight = (change.lines_added + change.lines_deleted) / 100; // Normalize
      existing.components.change_churn = Math.min(1.0,
        existing.components.change_churn + changeWeight);
      
      // Update owner proximity if author is a code owner
      const isOwner = this.isCodeOwner(change.author, filePath);
      if (isOwner) {
        existing.components.owner_proximity = Math.min(1.0,
          existing.components.owner_proximity + 0.2);
      }
      
      this.updateFilePriorScore(existing);
      this.filePriors.set(filePath, existing);
    }
  }

  private collectTransitiveDependencies(target: BuildTarget, collected: Set<string>): void {
    // Add target's sources
    for (const source of target.sources) {
      collected.add(source);
    }
    
    // Recursively collect dependencies (bounded to prevent infinite loops)
    if (collected.size < 1000) { // Safety limit
      for (const depId of target.deps) {
        const depTarget = this.buildTargets.get(depId);
        if (depTarget && !collected.has(depId)) {
          collected.add(depId);
          this.collectTransitiveDependencies(depTarget, collected);
        }
      }
    }
  }

  private createEmptyFilePrior(filePath: string): FilePrior {
    return {
      file_path: filePath,
      prior_score: 0,
      components: {
        failure_proximity: 0,
        change_churn: 0,
        owner_proximity: 0,
      },
      last_updated: new Date(),
      decay_factor: 1.0,
    };
  }

  private updateFilePriorScore(prior: FilePrior): void {
    // Weighted combination of components
    prior.prior_score = 
      0.5 * prior.components.failure_proximity +
      0.3 * prior.components.change_churn +
      0.2 * prior.components.owner_proximity;
      
    prior.last_updated = new Date();
  }

  private isCodeOwner(author: string, filePath: string): boolean {
    for (const owner of this.codeOwners) {
      if (this.matchesGlobPattern(filePath, owner.pattern)) {
        return owner.owners.includes(author);
      }
    }
    return false;
  }

  private matchesGlobPattern(filePath: string, pattern: string): boolean {
    // Simple glob matching - could be enhanced with a proper glob library
    const regexPattern = pattern
      .replace(/\*/g, '.*')
      .replace(/\?/g, '.');
    
    return new RegExp(`^${regexPattern}$`).test(filePath);
  }

  /**
   * Get statistics about the build/test priors system
   */
  getStats(): {
    targets: number;
    failures: number;
    changes: number;
    priors: number;
    recent_failures: number;
  } {
    const recentCutoff = new Date(Date.now() - 24 * 60 * 60 * 1000);
    const recentFailures = Array.from(this.testFailures.values())
      .flat()
      .filter(f => f.failure_time > recentCutoff).length;

    return {
      targets: this.buildTargets.size,
      failures: Array.from(this.testFailures.values()).reduce((sum, failures) => sum + failures.length, 0),
      changes: this.changeEvents.length,
      priors: this.filePriors.size,
      recent_failures: recentFailures,
    };
  }
}