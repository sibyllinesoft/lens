/**
 * Normalize helpers for consistent span computation
 * Single policy for all text processing across stages
 */

/**
 * Normalize Unicode text to NFC form to prevent combining-character drift
 * This ensures consistent span math across different Unicode representations
 */
export function normalizeUnicodeNFC(text: string): string {
  return text.normalize('NFC');
}

/**
 * Normalize line endings to \n for consistent counting
 * Leaves files on disk untouched - only normalizes in memory
 */
export function normalizeLineEndings(text: string): string {
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

/**
 * Apply all normalization steps for consistent text processing
 * This is the master normalization function that should be used at ingest
 */
export function normalizeText(text: string): string {
  return normalizeLineEndings(normalizeUnicodeNFC(text));
}

/**
 * Count Unicode code points for column positioning
 * Tabs count as 1 code point (consistent with tree-sitter)
 */
export function getCodePointColumn(text: string, byteOffset: number): number {
  // Convert byte offset to character offset, handling incomplete UTF-8 sequences
  const bytes = Buffer.from(text, 'utf8');
  
  // Ensure we don't slice in the middle of a multi-byte character
  let adjustedOffset = byteOffset;
  if (adjustedOffset > bytes.length) {
    adjustedOffset = bytes.length;
  }
  
  // Find the nearest valid UTF-8 character boundary at or before byteOffset
  while (adjustedOffset > 0) {
    try {
      const beforeBytes = bytes.slice(0, adjustedOffset);
      const beforeText = beforeBytes.toString('utf8');
      
      // Check if we get valid UTF-8 (no replacement chars)
      if (!beforeText.includes('\uFFFD')) {
        const normalizedText = normalizeText(beforeText);
        
        // Find the last newline to get column position
        const lastNewline = normalizedText.lastIndexOf('\n');
        const lineText = lastNewline === -1 ? normalizedText : normalizedText.slice(lastNewline + 1);
        
        // Count code points (not bytes)
        return Array.from(lineText).length;
      }
    } catch (e) {
      // Invalid UTF-8, try one byte earlier
    }
    adjustedOffset--;
  }
  
  return 0; // Fallback for position 0
}

/**
 * Get line number from byte offset (1-based)
 */
export function getLineFromByteOffset(text: string, byteOffset: number): number {
  // Convert byte offset to character offset, handling incomplete UTF-8 sequences
  const bytes = Buffer.from(text, 'utf8');
  
  // Ensure we don't slice in the middle of a multi-byte character
  let adjustedOffset = byteOffset;
  if (adjustedOffset > bytes.length) {
    adjustedOffset = bytes.length;
  }
  
  // Find the nearest valid UTF-8 character boundary at or before byteOffset
  while (adjustedOffset > 0) {
    try {
      const beforeBytes = bytes.slice(0, adjustedOffset);
      const beforeText = beforeBytes.toString('utf8');
      
      // Check if we get valid UTF-8 (no replacement chars)
      if (!beforeText.includes('\uFFFD')) {
        // Special handling: if we end with \r (part of \r\n), don't count it as a line break yet
        let textToCount = beforeText;
        if (beforeText.endsWith('\r') && adjustedOffset < bytes.length) {
          // Check if the next byte is \n to complete the \r\n sequence
          if (bytes[adjustedOffset] === 0x0A) { // \n
            // We're in the middle of \r\n, don't count this as a line break
            textToCount = beforeText.slice(0, -1); // Remove trailing \r
          }
        }
        
        const normalizedText = normalizeText(textToCount);
        
        // Count newlines + 1
        return (normalizedText.match(/\n/g) || []).length + 1;
      }
    } catch (e) {
      // Invalid UTF-8, try one byte earlier
    }
    adjustedOffset--;
  }
  
  return 1; // Fallback for line 1
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
  const lines = normalizeText(text).split('\n');
  
  // Convert to 0-based indexing
  const lineIndex = line - 1;
  
  if (lineIndex < 0 || lineIndex >= lines.length) {
    return '';
  }
  
  const targetLine = lines[lineIndex];
  if (!targetLine) {
    return '';
  }
  
  let startCol = Math.max(0, col - Math.floor(maxLength / 2));
  let endCol = Math.min(targetLine.length, startCol + maxLength);
  
  // If we're near the end of the line, align to show the end instead of centering
  const optimalEndStart = Math.max(0, targetLine.length - maxLength + 3); // +3 for ellipsis
  if (col >= optimalEndStart - 5 && targetLine.length > maxLength) {
    // Reserve space for leading ellipsis
    const textSpace = maxLength - 3;
    startCol = Math.max(0, targetLine.length - textSpace);
    endCol = targetLine.length;
  }
  
  let snippet = targetLine.slice(startCol, endCol);
  
  // Add ellipsis if truncated
  if (startCol > 0) {
    snippet = '...' + snippet;
  }
  // Only add trailing ellipsis if we actually truncated and have room
  if (endCol < targetLine.length && snippet.length < maxLength) {
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
  const lines = normalizeText(text).split('\n');
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
  
  const lines = normalizeText(text).split('\n');
  
  if (line > lines.length) {
    return { valid: false, error: `Line ${line} exceeds file length (${lines.length})` };
  }
  
  const targetLine = lines[line - 1];
  if (targetLine === undefined) {
    return { valid: false, error: `Line ${line} is undefined` };
  }
  // Empty lines are valid - they just have a max column of 0
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