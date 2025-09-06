/**
 * API-Evolution Mapping System
 * 
 * Two-layer mapping: (a) symbol lineage from structural diffs (rename/move/sig change)
 * and (b) usage rewrite rules mined from edit history (before→after patterns).
 * At query time: expand terms/symbols along lineage edges under tight budgets.
 * Project spans via revision line-maps, emit why+="evolution:{rename|move|sig}".
 * Recovers "where did Foo go?" queries without touching embeddings.
 * 
 * Gates: Success@10 +0.5 pp on "where is X" queries, zero span drift across HEAD↔SHA
 * Constraints: budget ≤2% query time, lineage maps for top packages
 */

import { LensTracer } from '../telemetry/tracer.js';
import type { SearchHit, SymbolCandidate, MatchReason } from './span_resolver/types.js';

export interface SymbolLineage {
  symbol_id: string;
  evolution_chain: EvolutionEvent[];
  current_location: SymbolLocation;
  historical_locations: SymbolLocation[];
  confidence: number;          // 0.0-1.0
  last_updated: Date;
}

export interface EvolutionEvent {
  type: 'rename' | 'move' | 'signature_change' | 'split' | 'merge';
  from_symbol: string;
  to_symbol: string;
  from_location: SymbolLocation;
  to_location: SymbolLocation;
  commit_sha: string;
  timestamp: Date;
  confidence: number;
  evidence: EvolutionEvidence[];
}

export interface EvolutionEvidence {
  type: 'structural_diff' | 'edit_pattern' | 'git_rename' | 'commit_message';
  data: any;
  confidence: number;
}

export interface SymbolLocation {
  repo_sha: string;
  file_path: string;
  line: number;
  col: number;
  span_len?: number;
  signature?: string;
}

export interface UsageRewriteRule {
  rule_id: string;
  pattern: string;             // Before pattern (regex)
  replacement: string;         // After pattern
  context: string;             // Surrounding context for disambiguation
  frequency: number;           // How often this rewrite appears
  confidence: number;          // Rule confidence (0.0-1.0)
  package_scope?: string;      // Scope to specific packages
  version_range?: string;      // Version range where rule applies
  examples: RewriteExample[];
}

export interface RewriteExample {
  before: string;
  after: string;
  file_path: string;
  commit_sha: string;
}

export interface RevisionLineMap {
  repo_sha: string;
  base_sha: string;
  file_path: string;
  line_mapping: LineMapping[];  // Maps line numbers between revisions
  created_at: Date;
  expires_at: Date;
}

export interface LineMapping {
  base_line: number;
  target_line: number;
  confidence: number;
  operation: 'unchanged' | 'moved' | 'modified' | 'added' | 'deleted';
}

export interface EvolutionMappingConfig {
  max_lineage_depth: number;
  max_rewrite_rules_per_symbol: number;
  max_query_time_budget_percent: number; // ≤2%
  line_map_ttl_hours: number;
  confidence_threshold: number;
  supported_packages: string[]; // Top packages only per constraints
  performance_targets: {
    success_at_10_improvement_pp: number; // ≥0.5pp
    zero_span_drift_tolerance: number;    // ≤0 for strict enforcement
  };
}

interface RewriteCandidate {
  pattern: string;
  replacement: string;
  context: string;
  frequency: number;
  confidence: number;
  package_scope?: string;
  examples: RewriteExample[];
}

export class EvolutionMappingSystem {
  private symbolLineages = new Map<string, SymbolLineage>();
  private rewriteRules = new Map<string, UsageRewriteRule[]>();
  private revisionLineMaps = new Map<string, RevisionLineMap>();
  private structuralDiffCache = new Map<string, any>();
  
  constructor(
    private config: EvolutionMappingConfig = {
      max_lineage_depth: 10,
      max_rewrite_rules_per_symbol: 20,
      max_query_time_budget_percent: 2,
      line_map_ttl_hours: 24,
      confidence_threshold: 0.7,
      supported_packages: [], // Configure with top packages
      performance_targets: {
        success_at_10_improvement_pp: 0.5,
        zero_span_drift_tolerance: 0,
      }
    }
  ) {}

  /**
   * Build symbol lineage from structural diffs between revisions
   */
  async buildSymbolLineage(
    repoSha: string,
    baseCommit: string,
    targetCommit: string
  ): Promise<void> {
    const span = LensTracer.createChildSpan('build_symbol_lineage', {
      'repo.sha': repoSha,
      'base.commit': baseCommit,
      'target.commit': targetCommit,
    });

    try {
      // Extract structural diffs
      const structuralDiff = await this.extractStructuralDiff(
        repoSha, 
        baseCommit, 
        targetCommit
      );

      // Process each changed file
      for (const fileDiff of structuralDiff.files) {
        await this.processFileDiff(repoSha, fileDiff, baseCommit, targetCommit);
      }

      // Build lineage chains
      await this.consolidateLineageChains(repoSha);

      span.setAttributes({
        'lineages.created': this.symbolLineages.size,
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
   * Mine usage rewrite rules from edit history
   */
  async mineUsageRewriteRules(
    repoSha: string,
    commitRange: string,
    packageScope?: string
  ): Promise<void> {
    const span = LensTracer.createChildSpan('mine_rewrite_rules', {
      'repo.sha': repoSha,
      'commit.range': commitRange,
      'package.scope': packageScope || 'all',
    });

    try {
      // Get commit history in range
      const commits = await this.getCommitHistory(repoSha, commitRange);
      
      const patterns = new Map<string, RewriteCandidate>();

      // Process each commit for edit patterns
      for (const commit of commits) {
        const editPatterns = await this.extractEditPatterns(repoSha, commit);
        
        for (const pattern of editPatterns) {
          // Skip if not in supported packages
          if (packageScope && !this.isInPackageScope(pattern, packageScope)) {
            continue;
          }

          const key = this.generatePatternKey(pattern);
          const existing = patterns.get(key);
          
          if (existing) {
            existing.frequency++;
            existing.examples.push(pattern.example);
          } else {
            patterns.set(key, {
              pattern: pattern.before,
              replacement: pattern.after,
              context: pattern.context,
              frequency: 1,
              confidence: pattern.confidence,
              package_scope: packageScope,
              examples: [pattern.example]
            });
          }
        }
      }

      // Convert to rewrite rules
      for (const [key, candidate] of patterns) {
        if (candidate.frequency >= 2 && candidate.confidence >= this.config.confidence_threshold) {
          const rule: UsageRewriteRule = {
            rule_id: key,
            pattern: candidate.pattern,
            replacement: candidate.replacement,
            context: candidate.context,
            frequency: candidate.frequency,
            confidence: candidate.confidence,
            package_scope: candidate.package_scope,
            examples: candidate.examples.slice(0, 5) // Limit examples
          };

          const symbolRules = this.rewriteRules.get(candidate.pattern) || [];
          if (symbolRules.length < this.config.max_rewrite_rules_per_symbol) {
            symbolRules.push(rule);
            this.rewriteRules.set(candidate.pattern, symbolRules);
          }
        }
      }

      span.setAttributes({
        'rules.mined': this.rewriteRules.size,
        'commits.processed': commits.length,
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
   * Expand query terms/symbols along lineage edges during query time
   */
  async expandWithEvolution(
    query: string,
    candidates: SymbolCandidate[],
    maxBudgetMs: number
  ): Promise<SymbolCandidate[]> {
    const span = LensTracer.createChildSpan('expand_with_evolution', {
      'query': query,
      'candidates.input': candidates.length,
      'budget.ms': maxBudgetMs
    });

    const startTime = performance.now();
    const expandedCandidates: SymbolCandidate[] = [...candidates];

    try {
      // Enforce budget constraint (≤2% query time)
      const budgetMs = Math.min(maxBudgetMs * this.config.max_query_time_budget_percent / 100, maxBudgetMs);
      
      // Extract symbols from query
      const querySymbols = this.extractSymbolsFromQuery(query);
      
      // Expand each symbol through lineage
      for (const symbol of querySymbols) {
        if (performance.now() - startTime > budgetMs) {
          break; // Respect time budget
        }

        const lineage = this.symbolLineages.get(symbol);
        if (lineage) {
          const evolutionCandidates = await this.expandFromLineage(lineage, candidates);
          expandedCandidates.push(...evolutionCandidates);
        }

        // Apply rewrite rules
        const rewriteRules = this.rewriteRules.get(symbol);
        if (rewriteRules) {
          const rewriteCandidates = await this.expandFromRewriteRules(
            symbol, 
            rewriteRules, 
            candidates
          );
          expandedCandidates.push(...rewriteCandidates);
        }
      }

      const elapsedMs = performance.now() - startTime;
      
      span.setAttributes({
        'candidates.output': expandedCandidates.length,
        'candidates.expanded': expandedCandidates.length - candidates.length,
        'symbols.expanded': querySymbols.length,
        'budget.used_ms': elapsedMs,
        'budget.within_limits': elapsedMs <= budgetMs,
        success: true
      });

      return expandedCandidates;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Project spans via revision line-maps and emit evolution why reasons
   */
  async projectSpansAcrossRevisions(
    hits: SearchHit[],
    sourceRevision: string,
    targetRevision: string
  ): Promise<SearchHit[]> {
    const span = LensTracer.createChildSpan('project_spans_across_revisions', {
      'hits.count': hits.length,
      'source.revision': sourceRevision,
      'target.revision': targetRevision
    });

    try {
      const projectedHits: SearchHit[] = [];

      for (const hit of hits) {
        const lineMap = await this.getRevisionLineMap(
          hit.file, 
          sourceRevision, 
          targetRevision
        );

        if (lineMap) {
          const projectedHit = await this.projectHit(hit, lineMap);
          if (projectedHit) {
            // Add evolution why reason
            const evolutionReason = this.determineEvolutionReason(hit, projectedHit);
            projectedHit.why.push(`evolution:${evolutionReason}` as MatchReason);
            projectedHits.push(projectedHit);
          }
        } else {
          // No line mapping available - pass through unchanged
          projectedHits.push(hit);
        }
      }

      // Verify zero span drift constraint
      const spanDrift = this.calculateSpanDrift(hits, projectedHits);
      if (spanDrift > this.config.performance_targets.zero_span_drift_tolerance) {
        console.warn(`Span drift detected: ${spanDrift} > ${this.config.performance_targets.zero_span_drift_tolerance}`);
      }

      span.setAttributes({
        'hits.projected': projectedHits.length,
        'span.drift': spanDrift,
        'drift.within_tolerance': spanDrift <= this.config.performance_targets.zero_span_drift_tolerance,
        success: true
      });

      return projectedHits;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false });
      throw error;
    } finally {
      span.end();
    }
  }

  // Implementation methods

  private async extractStructuralDiff(
    repoSha: string,
    baseCommit: string,
    targetCommit: string
  ): Promise<any> {
    // Placeholder for git diff + AST analysis
    // In production, this would use tree-sitter or similar to extract structural changes
    const cacheKey = `${repoSha}:${baseCommit}:${targetCommit}`;
    
    if (this.structuralDiffCache.has(cacheKey)) {
      return this.structuralDiffCache.get(cacheKey);
    }

    // Mock structural diff for now
    const structuralDiff = {
      files: [
        {
          path: 'example.ts',
          changes: [
            {
              type: 'rename',
              from: 'oldFunction',
              to: 'newFunction',
              line: 42,
              confidence: 0.9
            }
          ]
        }
      ]
    };

    this.structuralDiffCache.set(cacheKey, structuralDiff);
    return structuralDiff;
  }

  private async processFileDiff(
    repoSha: string,
    fileDiff: any,
    baseCommit: string,
    targetCommit: string
  ): Promise<void> {
    for (const change of fileDiff.changes) {
      if (change.type === 'rename' && change.confidence >= this.config.confidence_threshold) {
        const lineage: SymbolLineage = {
          symbol_id: change.from,
          evolution_chain: [{
            type: 'rename',
            from_symbol: change.from,
            to_symbol: change.to,
            from_location: {
              repo_sha: repoSha,
              file_path: fileDiff.path,
              line: change.line,
              col: 0
            },
            to_location: {
              repo_sha: repoSha,
              file_path: fileDiff.path,
              line: change.line,
              col: 0
            },
            commit_sha: targetCommit,
            timestamp: new Date(),
            confidence: change.confidence,
            evidence: [{
              type: 'structural_diff',
              data: change,
              confidence: change.confidence
            }]
          }],
          current_location: {
            repo_sha: repoSha,
            file_path: fileDiff.path,
            line: change.line,
            col: 0
          },
          historical_locations: [],
          confidence: change.confidence,
          last_updated: new Date()
        };

        this.symbolLineages.set(change.from, lineage);
      }
    }
  }

  private async consolidateLineageChains(repoSha: string): Promise<void> {
    // Consolidate separate evolution events into coherent lineage chains
    // This would implement graph traversal to connect related events
  }

  private async getCommitHistory(repoSha: string, range: string): Promise<any[]> {
    // Placeholder for git log parsing
    return [];
  }

  private async extractEditPatterns(repoSha: string, commit: any): Promise<any[]> {
    // Placeholder for edit pattern extraction
    return [];
  }

  private extractSymbolsFromQuery(query: string): string[] {
    // Basic symbol extraction - can be enhanced with proper parsing
    const symbols = query.match(/\b[A-Z][a-zA-Z0-9_]*\b/g) || [];
    return [...new Set(symbols)]; // Dedupe
  }

  private async expandFromLineage(
    lineage: SymbolLineage,
    candidates: SymbolCandidate[]
  ): Promise<SymbolCandidate[]> {
    const expanded: SymbolCandidate[] = [];

    for (const event of lineage.evolution_chain) {
      // Create candidate for evolved symbol location
      const candidate: SymbolCandidate = {
        file_path: event.to_location.file_path,
        score: 0.8 * event.confidence, // Adjust score by confidence
        match_reasons: ['symbol'] as MatchReason[],
        upstream_line: event.to_location.line,
        upstream_col: event.to_location.col
      };

      expanded.push(candidate);
    }

    return expanded;
  }

  private async expandFromRewriteRules(
    symbol: string,
    rules: UsageRewriteRule[],
    candidates: SymbolCandidate[]
  ): Promise<SymbolCandidate[]> {
    const expanded: SymbolCandidate[] = [];

    for (const rule of rules) {
      if (rule.confidence >= this.config.confidence_threshold) {
        // Apply rewrite rule to generate candidates
        // This would involve pattern matching and replacement
      }
    }

    return expanded;
  }

  private async getRevisionLineMap(
    filePath: string,
    sourceRevision: string,
    targetRevision: string
  ): Promise<RevisionLineMap | null> {
    const key = `${filePath}:${sourceRevision}:${targetRevision}`;
    const cached = this.revisionLineMaps.get(key);

    if (cached && cached.expires_at > new Date()) {
      return cached;
    }

    // Build new line map
    const lineMap: RevisionLineMap = {
      repo_sha: sourceRevision,
      base_sha: targetRevision,
      file_path: filePath,
      line_mapping: [], // Would be populated by git diff analysis
      created_at: new Date(),
      expires_at: new Date(Date.now() + this.config.line_map_ttl_hours * 3600 * 1000)
    };

    this.revisionLineMaps.set(key, lineMap);
    return lineMap;
  }

  private async projectHit(hit: SearchHit, lineMap: RevisionLineMap): Promise<SearchHit | null> {
    const mapping = lineMap.line_mapping.find(m => m.base_line === hit.line);
    if (!mapping) return null;

    return {
      ...hit,
      line: mapping.target_line,
      why: [...hit.why] // Will be augmented with evolution reason by caller
    };
  }

  private determineEvolutionReason(original: SearchHit, projected: SearchHit): string {
    if (original.file !== projected.file) return 'move';
    if (original.line !== projected.line) return 'move';
    return 'rename'; // Default
  }

  private calculateSpanDrift(original: SearchHit[], projected: SearchHit[]): number {
    // Calculate drift between original and projected spans
    // For now, simple metric - can be enhanced with more sophisticated analysis
    let drift = 0;
    
    for (let i = 0; i < Math.min(original.length, projected.length); i++) {
      const orig = original[i];
      const proj = projected[i];
      
      if (orig.line !== proj.line || orig.col !== proj.col) {
        drift++;
      }
    }

    return drift / original.length; // Return as ratio
  }


  private isInPackageScope(pattern: any, packageScope: string): boolean {
    // Implement package scope checking
    return true;
  }

  private generatePatternKey(pattern: any): string {
    return `${pattern.before}->${pattern.after}`;
  }

  /**
   * Get performance metrics for gate monitoring
   */
  getPerformanceMetrics() {
    return {
      lineages_tracked: this.symbolLineages.size,
      rewrite_rules_count: Array.from(this.rewriteRules.values()).reduce((acc, rules) => acc + rules.length, 0),
      line_maps_cached: this.revisionLineMaps.size,
      avg_lineage_depth: this.getAverageLineageDepth(),
      avg_rule_confidence: this.getAverageRuleConfidence(),
    };
  }

  private getAverageLineageDepth(): number {
    const depths = Array.from(this.symbolLineages.values()).map(l => l.evolution_chain.length);
    return depths.length > 0 ? depths.reduce((a, b) => a + b, 0) / depths.length : 0;
  }

  private getAverageRuleConfidence(): number {
    const rules = Array.from(this.rewriteRules.values()).flat();
    const confidences = rules.map(r => r.confidence);
    return confidences.length > 0 ? confidences.reduce((a, b) => a + b, 0) / confidences.length : 0;
  }
}