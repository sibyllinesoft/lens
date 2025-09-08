/**
 * Pooled Qrels Builder - Builds qrels from union of top-k across all systems
 * 
 * This implements the core pooled qrels methodology where relevance judgments
 * are built from the union of in-SLA top-k results across all participating
 * systems to avoid bias toward any single system.
 */

import { createReadStream, createWriteStream } from 'fs';
import { createInterface } from 'readline';
import { pipeline } from 'stream/promises';
import * as path from 'path';

export interface PooledQrelsConfig {
  suites: string[];  // ['coir', 'swe_verified', 'csn', 'cosqa', 'cp_regex']
  systems: string[]; // ['lens', 'bm25', 'bm25_prox', 'hybrid', 'sourcegraph']
  sla_ms: number;    // 150ms SLA enforcement
  top_k: number;     // Top-K results to include in pool (default: 50)
  min_agreement: number; // Minimum systems that must return result (default: 2)
  output_dir: string;
}

export interface SearchResult {
  query_id: string;
  system_id: string;
  rank: number;
  file: string;
  line: number;
  column: number;
  score: number;
  latency_ms: number;
  within_sla: boolean;
  snippet: string;
  why_tag: 'exact' | 'struct' | 'semantic' | 'mixed';
}

export interface QrelJudgment {
  query_id: string;
  file: string;
  line: number;
  column: number;
  relevance: number; // 0-3 scale: 0=irrelevant, 1=relevant, 2=highly_relevant, 3=perfect
  agreement_count: number; // How many systems returned this result
  contributing_systems: string[];
  avg_rank: number;
  avg_score: number;
  consensus_why_tag: string;
}

export class PooledQrelsBuilder {
  constructor(private config: PooledQrelsConfig) {}

  /**
   * Build pooled qrels from system results
   */
  async buildPooledQrels(): Promise<void> {
    console.log(`🏗️  Building pooled qrels for suites: ${this.config.suites.join(', ')}`);
    console.log(`📊 Systems: ${this.config.systems.join(', ')}`);
    console.log(`⏱️  SLA: ${this.config.sla_ms}ms`);
    
    for (const suite of this.config.suites) {
      console.log(`\n📋 Processing suite: ${suite}`);
      await this.buildQrelsForSuite(suite);
    }
    
    console.log(`\n✅ Pooled qrels generation complete`);
    console.log(`📁 Output directory: ${this.config.output_dir}`);
  }

  /**
   * Build qrels for a specific test suite
   */
  private async buildQrelsForSuite(suite: string): Promise<void> {
    const queryResults = new Map<string, SearchResult[]>();
    
    // Collect results from all systems for this suite
    for (const system of this.config.systems) {
      const resultsFile = path.join('runs', suite, `${system}_results.ndjson`);
      
      try {
        const results = await this.loadSystemResults(resultsFile);
        console.log(`📊 Loaded ${results.length} results from ${system}`);
        
        // Group by query_id
        for (const result of results) {
          if (!queryResults.has(result.query_id)) {
            queryResults.set(result.query_id, []);
          }
          queryResults.get(result.query_id)!.push(result);
        }
      } catch (error) {
        console.warn(`⚠️  Failed to load results for ${system}: ${error}`);
      }
    }
    
    // Build pooled judgments
    const qrels: QrelJudgment[] = [];
    
    for (const [queryId, results] of queryResults) {
      const pooledJudgments = this.buildPoolForQuery(queryId, results);
      qrels.push(...pooledJudgments);
    }
    
    // Write qrels file
    const outputFile = path.join(this.config.output_dir, `${suite}_pooled_qrels.json`);
    await this.writeQrelsFile(outputFile, qrels);
    
    // Generate qrels statistics
    const stats = this.generateQrelsStats(suite, qrels);
    const statsFile = path.join(this.config.output_dir, `${suite}_qrels_stats.json`);
    await this.writeStatsFile(statsFile, stats);
    
    console.log(`✅ Generated ${qrels.length} pooled judgments for ${suite}`);
  }

  /**
   * Build pooled judgments for a single query
   */
  private buildPoolForQuery(queryId: string, results: SearchResult[]): QrelJudgment[] {
    // Filter to in-SLA results only
    const inSlaResults = results.filter(r => r.within_sla && r.latency_ms <= this.config.sla_ms);
    
    // Group results by (file, line, column) to identify unique hits
    const hitGroups = new Map<string, SearchResult[]>();
    
    for (const result of inSlaResults) {
      const hitKey = `${result.file}:${result.line}:${result.column}`;
      if (!hitGroups.has(hitKey)) {
        hitGroups.set(hitKey, []);
      }
      hitGroups.get(hitKey)!.push(result);
    }
    
    const judgments: QrelJudgment[] = [];
    
    for (const [hitKey, hitResults] of hitGroups) {
      // Must have minimum agreement across systems
      if (hitResults.length < this.config.min_agreement) {
        continue;
      }
      
      // Only include hits from top-K of each system
      const validHits = hitResults.filter(r => r.rank <= this.config.top_k);
      if (validHits.length === 0) {
        continue;
      }
      
      // Calculate relevance based on system agreement and ranking
      const relevance = this.calculateRelevance(validHits);
      
      const judgment: QrelJudgment = {
        query_id: queryId,
        file: validHits[0].file,
        line: validHits[0].line,
        column: validHits[0].column,
        relevance,
        agreement_count: validHits.length,
        contributing_systems: [...new Set(validHits.map(h => h.system_id))],
        avg_rank: validHits.reduce((sum, h) => sum + h.rank, 0) / validHits.length,
        avg_score: validHits.reduce((sum, h) => sum + h.score, 0) / validHits.length,
        consensus_why_tag: this.getConsensusWhyTag(validHits)
      };
      
      judgments.push(judgment);
    }
    
    return judgments;
  }

  /**
   * Calculate relevance score based on system agreement and ranking
   */
  private calculateRelevance(hits: SearchResult[]): number {
    const systemCount = this.config.systems.length;
    const agreementRatio = hits.length / systemCount;
    const avgRank = hits.reduce((sum, h) => sum + h.rank, 0) / hits.length;
    
    // High agreement + low average rank = high relevance
    if (agreementRatio >= 0.8 && avgRank <= 3) return 3; // Perfect match
    if (agreementRatio >= 0.6 && avgRank <= 10) return 2; // Highly relevant
    if (agreementRatio >= 0.4 || avgRank <= 20) return 1; // Relevant
    return 0; // Not relevant (shouldn't happen due to filtering)
  }

  /**
   * Determine consensus why_tag across systems
   */
  private getConsensusWhyTag(hits: SearchResult[]): string {
    const tagCounts = new Map<string, number>();
    
    for (const hit of hits) {
      const count = tagCounts.get(hit.why_tag) || 0;
      tagCounts.set(hit.why_tag, count + 1);
    }
    
    // Return most common tag, or 'mixed' if tied
    let maxCount = 0;
    let consensusTag = 'mixed';
    
    for (const [tag, count] of tagCounts) {
      if (count > maxCount) {
        maxCount = count;
        consensusTag = tag;
      }
    }
    
    return consensusTag;
  }

  /**
   * Load system results from NDJSON file
   */
  private async loadSystemResults(filePath: string): Promise<SearchResult[]> {
    const results: SearchResult[] = [];
    
    const fileStream = createReadStream(filePath);
    const rl = createInterface({
      input: fileStream,
      crlfDelay: Infinity
    });
    
    for await (const line of rl) {
      try {
        const result = JSON.parse(line) as SearchResult;
        results.push(result);
      } catch (error) {
        console.warn(`⚠️  Failed to parse line: ${line}`);
      }
    }
    
    return results;
  }

  /**
   * Write qrels to JSON file
   */
  private async writeQrelsFile(filePath: string, qrels: QrelJudgment[]): Promise<void> {
    const writeStream = createWriteStream(filePath);
    writeStream.write(JSON.stringify(qrels, null, 2));
    writeStream.end();
    
    await new Promise((resolve, reject) => {
      writeStream.on('finish', resolve);
      writeStream.on('error', reject);
    });
  }

  /**
   * Generate statistics about qrels quality
   */
  private generateQrelsStats(suite: string, qrels: QrelJudgment[]) {
    const stats = {
      suite,
      total_judgments: qrels.length,
      queries_covered: new Set(qrels.map(q => q.query_id)).size,
      relevance_distribution: {
        perfect: qrels.filter(q => q.relevance === 3).length,
        highly_relevant: qrels.filter(q => q.relevance === 2).length,
        relevant: qrels.filter(q => q.relevance === 1).length,
        irrelevant: qrels.filter(q => q.relevance === 0).length
      },
      agreement_distribution: this.getAgreementDistribution(qrels),
      why_tag_distribution: this.getWhyTagDistribution(qrels),
      avg_judgments_per_query: qrels.length / new Set(qrels.map(q => q.query_id)).size,
      contributing_systems: [...new Set(qrels.flatMap(q => q.contributing_systems))],
      timestamp: new Date().toISOString()
    };
    
    return stats;
  }

  /**
   * Get distribution of system agreement counts
   */
  private getAgreementDistribution(qrels: QrelJudgment[]) {
    const distribution: Record<string, number> = {};
    
    for (const qrel of qrels) {
      const key = qrel.agreement_count.toString();
      distribution[key] = (distribution[key] || 0) + 1;
    }
    
    return distribution;
  }

  /**
   * Get distribution of why tags
   */
  private getWhyTagDistribution(qrels: QrelJudgment[]) {
    const distribution: Record<string, number> = {};
    
    for (const qrel of qrels) {
      const tag = qrel.consensus_why_tag;
      distribution[tag] = (distribution[tag] || 0) + 1;
    }
    
    return distribution;
  }

  /**
   * Write statistics to JSON file
   */
  private async writeStatsFile(filePath: string, stats: any): Promise<void> {
    const writeStream = createWriteStream(filePath);
    writeStream.write(JSON.stringify(stats, null, 2));
    writeStream.end();
    
    await new Promise((resolve, reject) => {
      writeStream.on('finish', resolve);
      writeStream.on('error', reject);
    });
  }
}

/**
 * Suite-specific qrels builders with specialized logic
 */

export class SWEBenchQrelsBuilder extends PooledQrelsBuilder {
  /**
   * SWE-bench specific relevance calculation based on witness coverage
   */
  protected calculateRelevance(hits: SearchResult[]): number {
    // For SWE-bench, relevance is based on witness coverage
    // This would need integration with witness validation system
    const avgScore = hits.reduce((sum, h) => sum + h.score, 0) / hits.length;
    const agreementRatio = hits.length / this.config.systems.length;
    
    // Higher threshold for SWE-bench due to precision requirements
    if (agreementRatio >= 0.8 && avgScore >= 0.8) return 3;
    if (agreementRatio >= 0.6 && avgScore >= 0.6) return 2; 
    if (agreementRatio >= 0.4 && avgScore >= 0.4) return 1;
    return 0;
  }
}

export class CoIRQrelsBuilder extends PooledQrelsBuilder {
  /**
   * CoIR-specific relevance calculation for multi-language code search
   */
  protected calculateRelevance(hits: SearchResult[]): number {
    const avgRank = hits.reduce((sum, h) => sum + h.rank, 0) / hits.length;
    const agreementRatio = hits.length / this.config.systems.length;
    
    // CoIR focuses on broad coverage, so slightly lower thresholds
    if (agreementRatio >= 0.6 && avgRank <= 5) return 3;
    if (agreementRatio >= 0.4 && avgRank <= 15) return 2;
    if (agreementRatio >= 0.3 || avgRank <= 30) return 1;
    return 0;
  }
}

/**
 * Factory function to create appropriate qrels builder for suite
 */
export function createQrelsBuilder(suite: string, config: PooledQrelsConfig): PooledQrelsBuilder {
  switch (suite) {
    case 'swe_verified':
      return new SWEBenchQrelsBuilder(config);
    case 'coir':
      return new CoIRQrelsBuilder(config);
    default:
      return new PooledQrelsBuilder(config);
  }
}

// CLI interface
if (require.main === module) {
  const config: PooledQrelsConfig = {
    suites: ['coir', 'swe_verified', 'csn', 'cosqa', 'cp_regex'],
    systems: ['lens', 'bm25', 'bm25_prox', 'hybrid', 'sourcegraph'],
    sla_ms: 150,
    top_k: 50,
    min_agreement: 2,
    output_dir: 'pool'
  };
  
  const builder = new PooledQrelsBuilder(config);
  builder.buildPooledQrels().catch(console.error);
}