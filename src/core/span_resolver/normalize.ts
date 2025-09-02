/**
 * Normalize helpers for consistent span computation
 * Single policy for all text processing across stages
 */

/**
 * Normalize line endings to \n for consistent counting
 * Leaves files on disk untouched - only normalizes in memory
 */
export function normalizeLineEndings(text: string): string {
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

/**
 * Count Unicode code points for column positioning
 * Tabs count as 1 code point (consistent with tree-sitter)
 */
export function getCodePointColumn(text: string, byteOffset: number): number {
  // Convert byte offset to character offset
  const beforeBytes = text.slice(0, byteOffset);
  const normalizedText = normalizeLineEndings(beforeBytes);
  
  // Find the last newline to get column position
  const lastNewline = normalizedText.lastIndexOf('\n');
  const lineText = lastNewline === -1 ? normalizedText : normalizedText.slice(lastNewline + 1);
  
  // Count code points (not bytes)
  return Array.from(lineText).length;
}

/**
 * Get line number from byte offset (1-based)
 */
export function getLineFromByteOffset(text: string, byteOffset: number): number {
  const beforeBytes = text.slice(0, byteOffset);
  const normalizedText = normalizeLineEndings(beforeBytes);
  
  // Count newlines + 1
  return (normalizedText.match(/\n/g) || []).length + 1;
}

/**
 * Extract snippet around a specific location
 */
export function extractSnippet(
  text: string, 
  line: number, 
  col: number, 
  maxLength: number = 100
): string {
  const lines = normalizeLineEndings(text).split('\n');
  
  // Convert to 0-based indexing
  const lineIndex = line - 1;
  
  if (lineIndex < 0 || lineIndex >= lines.length) {
    return '';
  }
  
  const targetLine = lines[lineIndex];
  if (!targetLine) {
    return '';
  }
  
  const startCol = Math.max(0, col - Math.floor(maxLength / 2));
  const endCol = Math.min(targetLine.length, startCol + maxLength);
  
  let snippet = targetLine.slice(startCol, endCol);
  
  // Add ellipsis if truncated
  if (startCol > 0) {
    snippet = '...' + snippet;
  }
  if (endCol < targetLine.length) {
    snippet = snippet + '...';
  }
  
  return snippet;
}

/**
 * Extract context lines (up to ±2 lines for eval tolerances)
 */
export function extractContext(
  text: string,
  line: number,
  contextSize: number = 2
): { context_before?: string | undefined; context_after?: string | undefined } {
  const lines = normalizeLineEndings(text).split('\n');
  const lineIndex = line - 1; // Convert to 0-based
  
  let context_before: string | undefined;
  let context_after: string | undefined;
  
  // Extract before context
  if (lineIndex > 0) {
    const beforeStart = Math.max(0, lineIndex - contextSize);
    const beforeLines = lines.slice(beforeStart, lineIndex);
    if (beforeLines.length > 0) {
      context_before = beforeLines.join('\n');
    }
  }
  
  // Extract after context  
  if (lineIndex < lines.length - 1) {
    const afterEnd = Math.min(lines.length, lineIndex + 1 + contextSize);
    const afterLines = lines.slice(lineIndex + 1, afterEnd);
    if (afterLines.length > 0) {
      context_after = afterLines.join('\n');
    }
  }
  
  return { context_before, context_after };
}

/**
 * Validate span coordinates are within bounds
 */
export function validateSpanBounds(
  text: string,
  line: number,
  col: number
): { valid: boolean; error?: string } {
  if (line < 1) {
    return { valid: false, error: 'Line must be >= 1' };
  }
  
  if (col < 0) {
    return { valid: false, error: 'Column must be >= 0' };
  }
  
  const lines = normalizeLineEndings(text).split('\n');
  
  if (line > lines.length) {
    return { valid: false, error: `Line ${line} exceeds file length (${lines.length})` };
  }
  
  const targetLine = lines[line - 1];
  if (!targetLine) {
    return { valid: false, error: `Line ${line} is empty or undefined` };
  }
  const maxCol = Array.from(targetLine).length;
  
  if (col > maxCol) {
    return { valid: false, error: `Column ${col} exceeds line length (${maxCol})` };
  }
  
  return { valid: true };
}

/**
 * Convert byte offset to line/col coordinates
 */
export function byteOffsetToLineCol(text: string, byteOffset: number): { line: number; col: number } {
  const line = getLineFromByteOffset(text, byteOffset);
  const col = getCodePointColumn(text, byteOffset);
  return { line, col };
}