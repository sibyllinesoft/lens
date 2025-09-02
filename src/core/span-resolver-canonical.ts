/**
 * Canonical SpanResolver Implementation
 * 
 * Per TODO.md: "Enforce one resolver: route ALL stages through SpanResolver(corpus_snapshot);
 * Stage-B must emit byte ranges → codepoint line/col via a single converter;
 * Stage-A's scanner must set BOTH byte_offset and (line,col,len) using the same newline/tabs policy;
 * Stage-C passes spans through untouched."
 *
 * Normalization Policy:
 * - Indexer writes LF only; runtime normalizes input to LF before mapping
 * - Columns count codepoints, and tabs advance by 1
 * - Store per-file line_starts_cp[] and byte↔cp maps to make conversions O(1)
 */

export interface SpanCoordinates {
  /** 1-indexed line number */
  line: number;
  /** 1-indexed column number (codepoint-based) */
  col: number;
  /** Length in codepoints (optional) */
  span_len?: number;
  /** Byte offset from start of file (optional) */
  byte_offset?: number;
  /** Byte length (optional) */
  byte_len?: number;
}

export interface FileSnapshot {
  filename: string;
  content: string;
  normalized_content: string;
  line_starts_cp: number[];
  byte_to_cp_map: Map<number, number>;
  cp_to_byte_map: Map<number, number>;
  hash: string;
}

export interface SpanValidationResult {
  valid: boolean;
  error_reason?: string;
  error_details?: string;
  canonical_coordinates?: SpanCoordinates;
}

/**
 * Canonical SpanResolver - Single source of truth for all span coordinate operations
 */
export class SpanResolverCanonical {
  private fileSnapshots = new Map<string, FileSnapshot>();
  private corpusDir: string;

  constructor(corpusDir: string = './indexed-content') {
    this.corpusDir = corpusDir;
  }

  /**
   * Load and normalize a file into the resolver
   */
  async loadFile(filename: string): Promise<FileSnapshot | null> {
    if (this.fileSnapshots.has(filename)) {
      return this.fileSnapshots.get(filename)!;
    }

    try {
      const fs = await import('fs');
      const path = await import('path');
      const crypto = await import('crypto');

      const filePath = path.join(this.corpusDir, filename);
      if (!fs.existsSync(filePath)) {
        return null;
      }

      const rawContent = fs.readFileSync(filePath, 'utf-8');
      const normalizedContent = this.normalizeContent(rawContent);
      
      const snapshot: FileSnapshot = {
        filename,
        content: rawContent,
        normalized_content: normalizedContent,
        line_starts_cp: this.buildLineStartsMap(normalizedContent),
        byte_to_cp_map: new Map(),
        cp_to_byte_map: new Map(),
        hash: crypto.createHash('sha256').update(normalizedContent).digest('hex').slice(0, 16)
      };

      this.buildByteCodepointMaps(snapshot);
      this.fileSnapshots.set(filename, snapshot);

      return snapshot;
    } catch (error) {
      console.warn(`Failed to load file ${filename}:`, error);
      return null;
    }
  }

  /**
   * Normalize content per TODO.md policy: CRLF→LF, tabs=1
   */
  private normalizeContent(content: string): string {
    // Convert CRLF to LF
    let normalized = content.replace(/\r\n/g, '\n');
    
    // Convert standalone CR to LF
    normalized = normalized.replace(/\r/g, '\n');
    
    // Note: tabs advance by 1 - no content conversion needed,
    // just consistent counting in coordinate calculations
    
    return normalized;
  }

  /**
   * Build line starts map for O(1) line/col conversion
   */
  private buildLineStartsMap(content: string): number[] {
    const lineStarts = [0]; // Line 1 starts at codepoint 0
    const codepoints = Array.from(content);

    for (let i = 0; i < codepoints.length; i++) {
      if (codepoints[i] === '\n') {
        lineStarts.push(i + 1); // Next line starts after the newline
      }
    }

    return lineStarts;
  }

  /**
   * Build bidirectional byte ↔ codepoint maps for O(1) conversion
   */
  private buildByteCodepointMaps(snapshot: FileSnapshot): void {
    const content = snapshot.normalized_content;
    const buffer = Buffer.from(content, 'utf-8');
    const codepoints = Array.from(content);

    let byteIndex = 0;
    for (let cpIndex = 0; cpIndex < codepoints.length; cpIndex++) {
      const codepoint = codepoints[cpIndex];
      const cpByteLength = Buffer.byteLength(codepoint, 'utf-8');

      // Map byte index to codepoint index
      snapshot.byte_to_cp_map.set(byteIndex, cpIndex);
      
      // Map codepoint index to byte index
      snapshot.cp_to_byte_map.set(cpIndex, byteIndex);

      byteIndex += cpByteLength;
    }

    // Add end-of-file mappings
    snapshot.byte_to_cp_map.set(buffer.length, codepoints.length);
    snapshot.cp_to_byte_map.set(codepoints.length, buffer.length);
  }

  /**
   * Convert byte offset to line/col coordinates (codepoint-based)
   */
  async byteOffsetToLineCol(filename: string, byteOffset: number): Promise<SpanCoordinates | null> {
    const snapshot = await this.loadFile(filename);
    if (!snapshot) return null;

    // Convert byte offset to codepoint index
    let cpIndex = snapshot.byte_to_cp_map.get(byteOffset);
    if (cpIndex === undefined) {
      // Find closest byte index
      const byteIndices = Array.from(snapshot.byte_to_cp_map.keys()).sort((a, b) => a - b);
      const closestByteIndex = byteIndices.reduce((prev, curr) => 
        Math.abs(curr - byteOffset) < Math.abs(prev - byteOffset) ? curr : prev
      );
      cpIndex = snapshot.byte_to_cp_map.get(closestByteIndex);
      if (cpIndex === undefined) return null;
    }

    return this.codepointIndexToLineCol(snapshot, cpIndex);
  }

  /**
   * Convert line/col coordinates to byte offset
   */
  async lineColToByteOffset(filename: string, line: number, col: number): Promise<number | null> {
    const snapshot = await this.loadFile(filename);
    if (!snapshot) return null;

    const cpIndex = this.lineColToCodepointIndex(snapshot, line, col);
    if (cpIndex === null) return null;

    return snapshot.cp_to_byte_map.get(cpIndex) || null;
  }

  /**
   * Convert codepoint index to line/col (1-indexed)
   */
  private codepointIndexToLineCol(snapshot: FileSnapshot, cpIndex: number): SpanCoordinates {
    const lineStarts = snapshot.line_starts_cp;

    // Find line number (binary search)
    let line = 1;
    for (let i = lineStarts.length - 1; i >= 0; i--) {
      if (cpIndex >= lineStarts[i]) {
        line = i + 1;
        break;
      }
    }

    // Calculate column (codepoint-based, tabs count as 1)
    const lineStart = lineStarts[line - 1];
    const col = cpIndex - lineStart + 1; // +1 for 1-indexed

    return { line, col };
  }

  /**
   * Convert line/col (1-indexed) to codepoint index
   */
  private lineColToCodepointIndex(snapshot: FileSnapshot, line: number, col: number): number | null {
    if (line < 1 || line > snapshot.line_starts_cp.length) {
      return null;
    }

    const lineStart = snapshot.line_starts_cp[line - 1];
    const cpIndex = lineStart + (col - 1); // -1 for 1-indexed

    // Validate bounds
    const maxCpIndex = Array.from(snapshot.normalized_content).length;
    if (cpIndex < 0 || cpIndex > maxCpIndex) {
      return null;
    }

    return cpIndex;
  }

  /**
   * Validate span coordinates against file content
   */
  async validateSpan(filename: string, coordinates: SpanCoordinates): Promise<SpanValidationResult> {
    const snapshot = await this.loadFile(filename);
    if (!snapshot) {
      return {
        valid: false,
        error_reason: 'CANDIDATE_MISSING',
        error_details: `File not found: ${filename}`
      };
    }

    const { line, col, span_len, byte_offset } = coordinates;
    const lines = snapshot.normalized_content.split('\n');

    // Validate line bounds
    if (line < 1 || line > lines.length) {
      return {
        valid: false,
        error_reason: 'OOB',
        error_details: `Line ${line} out of bounds (1-${lines.length})`
      };
    }

    const targetLine = lines[line - 1];
    const lineCodepoints = Array.from(targetLine);

    // Validate column bounds (allow end-of-line)
    if (col < 1 || col > lineCodepoints.length + 1) {
      return {
        valid: false,
        error_reason: 'OOB',
        error_details: `Column ${col} out of bounds (1-${lineCodepoints.length + 1}) for line ${line}`
      };
    }

    // Cross-validate byte offset if provided
    if (byte_offset !== undefined) {
      const expectedByteOffset = await this.lineColToByteOffset(filename, line, col);
      if (expectedByteOffset !== byte_offset) {
        return {
          valid: false,
          error_reason: 'BYTE_VS_CP',
          error_details: `Byte offset mismatch: expected ${expectedByteOffset}, got ${byte_offset}`
        };
      }
    }

    return {
      valid: true,
      canonical_coordinates: {
        line,
        col,
        span_len,
        byte_offset: byte_offset || await this.lineColToByteOffset(filename, line, col) || undefined
      }
    };
  }

  /**
   * Stage-A Scanner Interface: Generate canonical spans
   * Sets BOTH byte_offset and (line,col,len) using same policy
   */
  async generateCanonicalSpan(
    filename: string, 
    match: { line: number; col: number; length?: number } | { byte_offset: number; byte_length?: number }
  ): Promise<SpanCoordinates | null> {
    const snapshot = await this.loadFile(filename);
    if (!snapshot) return null;

    let coordinates: SpanCoordinates;

    if ('line' in match) {
      // Line/col input - calculate byte offset
      const byteOffset = await this.lineColToByteOffset(filename, match.line, match.col);
      coordinates = {
        line: match.line,
        col: match.col,
        span_len: match.length,
        byte_offset: byteOffset || undefined
      };
    } else {
      // Byte offset input - calculate line/col
      const lineCol = await this.byteOffsetToLineCol(filename, match.byte_offset);
      if (!lineCol) return null;

      coordinates = {
        line: lineCol.line,
        col: lineCol.col,
        span_len: match.byte_length,
        byte_offset: match.byte_offset
      };
    }

    // Validate the generated coordinates
    const validation = await this.validateSpan(filename, coordinates);
    return validation.valid ? coordinates : null;
  }

  /**
   * Stage-B Interface: Convert byte ranges to codepoint line/col
   */
  async convertByteRangeToLineCol(
    filename: string, 
    byteOffset: number, 
    byteLength?: number
  ): Promise<SpanCoordinates | null> {
    return this.generateCanonicalSpan(filename, { byte_offset: byteOffset, byte_length: byteLength });
  }

  /**
   * Stage-C Interface: Pass spans through untouched (validation only)
   */
  async passthrough(filename: string, coordinates: SpanCoordinates): Promise<SpanCoordinates | null> {
    const validation = await this.validateSpan(filename, coordinates);
    return validation.valid ? coordinates : null;
  }

  /**
   * Tree-sitter/LSIF Range Converter
   * Both produce byte offsets; convert with the same byte→cp table
   */
  async convertASTRange(
    filename: string, 
    range: { start_byte: number; end_byte: number }
  ): Promise<SpanCoordinates | null> {
    const startCoords = await this.byteOffsetToLineCol(filename, range.start_byte);
    if (!startCoords) return null;

    const byteLength = range.end_byte - range.start_byte;
    
    return {
      line: startCoords.line,
      col: startCoords.col,
      byte_offset: range.start_byte,
      byte_len: byteLength
    };
  }

  /**
   * Get file snapshot for debugging
   */
  getSnapshot(filename: string): FileSnapshot | null {
    return this.fileSnapshots.get(filename) || null;
  }

  /**
   * Clear cache for testing
   */
  clearCache(): void {
    this.fileSnapshots.clear();
  }

  /**
   * Get corpus statistics
   */
  getStats(): { loaded_files: number; total_codepoints: number; total_bytes: number } {
    let totalCodepoints = 0;
    let totalBytes = 0;

    for (const snapshot of this.fileSnapshots.values()) {
      totalCodepoints += Array.from(snapshot.normalized_content).length;
      totalBytes += Buffer.from(snapshot.normalized_content, 'utf-8').length;
    }

    return {
      loaded_files: this.fileSnapshots.size,
      total_codepoints: totalCodepoints,
      total_bytes: totalBytes
    };
  }
}