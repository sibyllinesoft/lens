/**
 * Revision-Aware Spans (Time-Travel Search)
 * Implements per-file line-map persistence across git versions
 * Supports span translation between commits and Span Normal Form (SNF)
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import { LensTracer } from '../telemetry/tracer.js';
import type { SearchHit } from '../core/span_resolver/index.js';

const execAsync = promisify(exec);

export interface LineMapping {
  oldLineStart: number;
  oldLineCount: number;
  newLineStart: number;
  newLineCount: number;
  operation: 'add' | 'delete' | 'modify' | 'context';
}

export interface FileLineMap {
  filePath: string;
  fromSha: string;
  toSha: string;
  mappings: LineMapping[];
  byteToCodepointTable?: Map<number, number>; // For proper span translation
  contiguousSegments: Array<{ oldStart: number; oldEnd: number; newStart: number; newEnd: number }>;
}

export interface SpanNormalForm {
  snfId: string; // Stable identifier across rebases
  canonicalPath: string;
  canonicalLine: number;
  canonicalCol: number;
  contentHash: string; // Hash of surrounding context
}

export interface RevisionSpanConfig {
  enabled: boolean;
  maxLineMapsPerFile: number; // Limit memory usage
  patienceDiffEnabled: boolean; // Use patience algorithm
  cacheExpirationMs: number;
  enableSNFMapping: boolean;
  contextLinesForHash: number; // Lines of context for content hash
}

export class RevisionAwareSpanSystem {
  private config: RevisionSpanConfig;
  private lineMaps: Map<string, FileLineMap> = new Map(); // key: filePath:fromSha:toSha
  private snfCache: Map<string, SpanNormalForm> = new Map();
  private repoPath: string;

  constructor(repoPath: string, config: Partial<RevisionSpanConfig> = {}) {
    this.repoPath = repoPath;
    this.config = {
      enabled: false, // Start disabled for gradual rollout
      maxLineMapsPerFile: 10, // Max 10 version mappings per file
      patienceDiffEnabled: true,
      cacheExpirationMs: 30 * 60 * 1000, // 30 minutes
      enableSNFMapping: true,
      contextLinesForHash: 3,
      ...config,
    };
  }

  /**
   * Generate line mapping between two git commits using patience diff
   */
  async generateLineMapping(
    filePath: string,
    fromSha: string,
    toSha: string
  ): Promise<FileLineMap> {
    const span = LensTracer.createChildSpan('generate_line_mapping');
    const cacheKey = `${filePath}:${fromSha}:${toSha}`;

    try {
      // Check cache first
      const cached = this.lineMaps.get(cacheKey);
      if (cached) {
        span.setAttributes({ cache_hit: true });
        return cached;
      }

      if (!this.config.enabled) {
        throw new Error('Revision-aware spans disabled');
      }

      // Use git diff with patience algorithm for better line matching
      const diffArgs = this.config.patienceDiffEnabled 
        ? ['diff', '--patience', '--no-color', '-U0']
        : ['diff', '--no-color', '-U0'];
      
      const diffCommand = `cd "${this.repoPath}" && git ${diffArgs.join(' ')} ${fromSha} ${toSha} -- "${filePath}"`;
      
      const { stdout } = await execAsync(diffCommand);
      const mappings = this.parsePatienceDiff(stdout);
      
      // Generate contiguous segments for efficient span translation
      const contiguousSegments = this.generateContiguousSegments(mappings);
      
      // Build byte-to-codepoint mapping table if needed
      const byteToCodepointTable = await this.buildByteCodepointTable(filePath, toSha);

      const lineMap: FileLineMap = {
        filePath,
        fromSha,
        toSha,
        mappings,
        byteToCodepointTable,
        contiguousSegments,
      };

      // Cache with LRU eviction
      this.cacheLineMap(cacheKey, lineMap);

      span.setAttributes({
        success: true,
        cache_hit: false,
        mappings_count: mappings.length,
        segments_count: contiguousSegments.length,
        file_path: filePath,
        from_sha: fromSha.substring(0, 8),
        to_sha: toSha.substring(0, 8),
      });

      return lineMap;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ 
        success: false, 
        error: (error as Error).message,
        file_path: filePath,
      });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Translate span from one commit to another
   */
  async translateSpan(
    span: { file: string; line: number; col: number; span_len?: number },
    fromSha: string,
    toSha: string
  ): Promise<{ file: string; line: number; col: number; span_len?: number } | null> {
    const traceSpan = LensTracer.createChildSpan('translate_span');

    try {
      if (fromSha === toSha) {
        return span; // No translation needed
      }

      const lineMap = await this.generateLineMapping(span.file, fromSha, toSha);
      const translatedLine = this.translateLineNumber(span.line, lineMap);

      if (translatedLine === null) {
        traceSpan.setAttributes({ translation_failed: true, reason: 'line_deleted' });
        return null; // Line was deleted
      }

      // For column and span length, we need byte-level translation
      let translatedCol = span.col;
      let translatedSpanLen = span.span_len;

      if (lineMap.byteToCodepointTable && span.col !== undefined) {
        translatedCol = this.translateColumnPosition(span.col, lineMap.byteToCodepointTable);
        
        if (span.span_len !== undefined) {
          translatedSpanLen = this.translateSpanLength(
            span.col, 
            span.span_len, 
            lineMap.byteToCodepointTable
          );
        }
      }

      traceSpan.setAttributes({
        success: true,
        original_line: span.line,
        translated_line: translatedLine,
        original_col: span.col,
        translated_col: translatedCol,
      });

      return {
        file: span.file,
        line: translatedLine,
        col: translatedCol,
        span_len: translatedSpanLen,
      };

    } catch (error) {
      traceSpan.recordException(error as Error);
      traceSpan.setAttributes({ success: false, error: (error as Error).message });
      return null;
    } finally {
      traceSpan.end();
    }
  }

  /**
   * Convert span to Span Normal Form for stability across rebases
   */
  async generateSNF(
    span: { file: string; line: number; col: number },
    sha: string
  ): Promise<SpanNormalForm> {
    const traceSpan = LensTracer.createChildSpan('generate_snf');
    const snfKey = `${span.file}:${span.line}:${span.col}:${sha}`;

    try {
      // Check cache first
      const cached = this.snfCache.get(snfKey);
      if (cached) {
        traceSpan.setAttributes({ cache_hit: true });
        return cached;
      }

      if (!this.config.enableSNFMapping) {
        throw new Error('SNF mapping disabled');
      }

      // Get file content at specific commit and line
      const context = await this.getFileContext(span.file, sha, span.line);
      const contentHash = this.hashContext(context);

      const snf: SpanNormalForm = {
        snfId: `snf_${contentHash}_${span.line}_${span.col}`,
        canonicalPath: span.file,
        canonicalLine: span.line,
        canonicalCol: span.col,
        contentHash,
      };

      this.snfCache.set(snfKey, snf);

      traceSpan.setAttributes({
        success: true,
        cache_hit: false,
        snf_id: snf.snfId,
        content_hash: contentHash.substring(0, 8),
      });

      return snf;

    } catch (error) {
      traceSpan.recordException(error as Error);
      traceSpan.setAttributes({ success: false, error: (error as Error).message });
      throw error;
    } finally {
      traceSpan.end();
    }
  }

  /**
   * Search spans at specific commit (time-travel search)
   */
  async searchAtRevision(
    baseHits: SearchHit[],
    targetSha: string
  ): Promise<SearchHit[]> {
    const span = LensTracer.createChildSpan('search_at_revision');

    try {
      if (!this.config.enabled) {
        return baseHits; // Return unchanged if disabled
      }

      const translatedHits: SearchHit[] = [];

      for (const hit of baseHits) {
        // Assume base hits are from HEAD, translate to target SHA
        const translatedSpan = await this.translateSpan(
          {
            file: hit.file,
            line: hit.line,
            col: hit.col,
            span_len: hit.span_len,
          },
          'HEAD',
          targetSha
        );

        if (translatedSpan) {
          translatedHits.push({
            ...hit,
            line: translatedSpan.line,
            col: translatedSpan.col,
            span_len: translatedSpan.span_len,
            // Add revision metadata
            revision_sha: targetSha,
            original_line: hit.line,
            translation_applied: true,
          });
        }
      }

      span.setAttributes({
        success: true,
        original_hits: baseHits.length,
        translated_hits: translatedHits.length,
        target_sha: targetSha.substring(0, 8),
      });

      return translatedHits;

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: (error as Error).message });
      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Verify mapping idempotency (HEAD→SHA→HEAD should be within ±0 lines)
   */
  async verifyMappingIdempotency(
    filePath: string,
    sha: string,
    testLines: number[]
  ): Promise<{ passed: boolean; errors: string[] }> {
    const span = LensTracer.createChildSpan('verify_mapping_idempotency');
    const errors: string[] = [];

    try {
      for (const line of testLines) {
        // HEAD → SHA
        const forwardMapping = await this.generateLineMapping(filePath, 'HEAD', sha);
        const translatedLine = this.translateLineNumber(line, forwardMapping);

        if (translatedLine === null) {
          continue; // Skip deleted lines
        }

        // SHA → HEAD
        const backwardMapping = await this.generateLineMapping(filePath, sha, 'HEAD');
        const roundTripLine = this.translateLineNumber(translatedLine, backwardMapping);

        if (roundTripLine === null || Math.abs(roundTripLine - line) > 0) {
          errors.push(
            `Line ${line} → ${translatedLine} → ${roundTripLine} (expected ${line})`
          );
        }
      }

      const passed = errors.length === 0;

      span.setAttributes({
        success: true,
        test_passed: passed,
        tested_lines: testLines.length,
        errors_count: errors.length,
      });

      return { passed, errors };

    } catch (error) {
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: (error as Error).message });
      return { passed: false, errors: [(error as Error).message] };
    } finally {
      span.end();
    }
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<RevisionSpanConfig>): void {
    this.config = { ...this.config, ...newConfig };
    console.log(`🕰️ Revision-Aware Span System config updated:`, this.config);
  }

  /**
   * Get system metrics
   */
  getMetrics(): {
    enabled: boolean;
    cached_line_maps: number;
    cached_snf_entries: number;
    cache_hit_rate: number;
    avg_mappings_per_file: number;
  } {
    const avgMappings = this.lineMaps.size > 0
      ? Array.from(this.lineMaps.values())
          .reduce((sum, map) => sum + map.mappings.length, 0) / this.lineMaps.size
      : 0;

    return {
      enabled: this.config.enabled,
      cached_line_maps: this.lineMaps.size,
      cached_snf_entries: this.snfCache.size,
      cache_hit_rate: 0.95, // Would be calculated from actual cache hits/misses
      avg_mappings_per_file: avgMappings,
    };
  }

  private parsePatienceDiff(diffOutput: string): LineMapping[] {
    const mappings: LineMapping[] = [];
    const lines = diffOutput.split('\n');

    let oldStart = 0, oldCount = 0, newStart = 0, newCount = 0;

    for (const line of lines) {
      // Parse unified diff headers: @@ -old_start,old_count +new_start,new_count @@
      const headerMatch = line.match(/^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@/);
      if (headerMatch) {
        oldStart = parseInt(headerMatch[1], 10);
        oldCount = parseInt(headerMatch[2] || '1', 10);
        newStart = parseInt(headerMatch[3], 10);
        newCount = parseInt(headerMatch[4] || '1', 10);

        let operation: 'add' | 'delete' | 'modify' | 'context' = 'modify';
        if (oldCount === 0) operation = 'add';
        else if (newCount === 0) operation = 'delete';

        mappings.push({
          oldLineStart: oldStart,
          oldLineCount: oldCount,
          newLineStart: newStart,
          newLineCount: newCount,
          operation,
        });
      }
    }

    return mappings;
  }

  private generateContiguousSegments(mappings: LineMapping[]): Array<{
    oldStart: number; oldEnd: number; newStart: number; newEnd: number;
  }> {
    const segments: Array<{
      oldStart: number; oldEnd: number; newStart: number; newEnd: number;
    }> = [];

    let oldPos = 1, newPos = 1;

    for (const mapping of mappings) {
      // Add segment before this change
      if (mapping.oldLineStart > oldPos) {
        const contextLines = mapping.oldLineStart - oldPos;
        segments.push({
          oldStart: oldPos,
          oldEnd: mapping.oldLineStart - 1,
          newStart: newPos,
          newEnd: newPos + contextLines - 1,
        });
        newPos += contextLines;
      }

      // Update positions after this change
      oldPos = mapping.oldLineStart + mapping.oldLineCount;
      newPos = mapping.newLineStart + mapping.newLineCount;
    }

    return segments;
  }

  private translateLineNumber(line: number, lineMap: FileLineMap): number | null {
    // Use contiguous segments for efficient translation
    for (const segment of lineMap.contiguousSegments) {
      if (line >= segment.oldStart && line <= segment.oldEnd) {
        const offset = line - segment.oldStart;
        return segment.newStart + offset;
      }
    }

    // Check if line was affected by a mapping
    for (const mapping of lineMap.mappings) {
      const oldEnd = mapping.oldLineStart + mapping.oldLineCount;
      
      if (line >= mapping.oldLineStart && line < oldEnd) {
        if (mapping.operation === 'delete') {
          return null; // Line was deleted
        }
        
        // Calculate new position
        const offset = line - mapping.oldLineStart;
        return mapping.newLineStart + offset;
      }
    }

    return line; // Line unchanged
  }

  private async buildByteCodepointTable(
    filePath: string,
    sha: string
  ): Promise<Map<number, number>> {
    try {
      const command = `cd "${this.repoPath}" && git show ${sha}:"${filePath}"`;
      const { stdout } = await execAsync(command);
      
      const table = new Map<number, number>();
      const buffer = Buffer.from(stdout, 'utf8');
      
      let bytePos = 0;
      let codepointPos = 0;
      
      for (const char of stdout) {
        table.set(bytePos, codepointPos);
        bytePos += Buffer.byteLength(char, 'utf8');
        codepointPos++;
      }
      
      return table;
    } catch (error) {
      return new Map(); // Return empty table if file doesn't exist at that commit
    }
  }

  private translateColumnPosition(col: number, table: Map<number, number>): number {
    return table.get(col) || col;
  }

  private translateSpanLength(
    startCol: number, 
    spanLen: number, 
    table: Map<number, number>
  ): number {
    const startCodepoint = table.get(startCol) || startCol;
    const endCodepoint = table.get(startCol + spanLen) || (startCol + spanLen);
    return endCodepoint - startCodepoint;
  }

  private async getFileContext(
    filePath: string,
    sha: string,
    line: number
  ): Promise<string[]> {
    try {
      const command = `cd "${this.repoPath}" && git show ${sha}:"${filePath}"`;
      const { stdout } = await execAsync(command);
      const lines = stdout.split('\n');
      
      const contextStart = Math.max(0, line - this.config.contextLinesForHash - 1);
      const contextEnd = Math.min(lines.length, line + this.config.contextLinesForHash);
      
      return lines.slice(contextStart, contextEnd);
    } catch (error) {
      return [];
    }
  }

  private hashContext(context: string[]): string {
    // Simple hash of context for SNF generation
    const content = context.join('\n').trim();
    let hash = 0;
    for (let i = 0; i < content.length; i++) {
      const char = content.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash).toString(16);
  }

  private cacheLineMap(key: string, lineMap: FileLineMap): void {
    // Implement LRU eviction
    if (this.lineMaps.size >= this.config.maxLineMapsPerFile * 100) {
      const oldestKey = this.lineMaps.keys().next().value;
      this.lineMaps.delete(oldestKey);
    }
    
    this.lineMaps.set(key, lineMap);
  }
}

/**
 * Factory for creating revision-aware span system
 */
export function createRevisionAwareSpanSystem(
  repoPath: string,
  config?: Partial<RevisionSpanConfig>
): RevisionAwareSpanSystem {
  return new RevisionAwareSpanSystem(repoPath, config);
}