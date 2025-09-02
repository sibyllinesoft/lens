export class SpanResolver {
    constructor(content, normalizeLineEndings = true) {
        this.normalizeLineEndings = normalizeLineEndings;
        // Conditionally normalize CRLF to LF
        this.text = normalizeLineEndings ? content.replace(/\r\n/g, '\n') : content;
        // Build line offset cache for efficient line/col conversion
        this.lineOffsets = [0]; // Line 1 starts at offset 0
        for (let i = 0; i < this.text.length; i++) {
            if (this.text[i] === '\n') {
                this.lineOffsets.push(i + 1);
            }
        }
    }
    getText() {
        return this.text;
    }
    byteToLineCol(byteOffset) {
        // Clamp to valid range
        const clampedOffset = Math.max(0, Math.min(byteOffset, this.text.length));
        // Handle empty file case
        if (this.text.length === 0) {
            return { line: 1, col: 1 };
        }
        // Find the line containing the byte offset
        let line = 1;
        let lineStart = 0;
        for (let i = this.lineOffsets.length - 1; i >= 0; i--) {
            const offset = this.lineOffsets[i];
            if (offset !== undefined && clampedOffset >= offset) {
                line = i + 1;
                lineStart = offset;
                break;
            }
        }
        // Calculate column within the line
        const colOffset = clampedOffset - lineStart;
        // For simple cases (ASCII text), column is just offset + 1
        // For Unicode, we need to count code points in the line content
        const lineContent = this.text.substring(lineStart, lineStart + colOffset);
        const col = Array.from(lineContent).length + 1;
        return { line, col };
    }
    resolveSpan(startOffset, endOffset) {
        const start = this.byteToLineCol(startOffset);
        const end = this.byteToLineCol(endOffset);
        return { start, end };
    }
}
