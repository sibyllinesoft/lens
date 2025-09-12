import { describe, it, expect } from 'bun:test';
import { SpanResolver } from './span_resolver';

describe('SpanResolver', () => {
  describe('text normalization', () => {
    it('should normalize CRLF line endings to LF', () => {
      const content = "line1\r\nline2\r\nline3";
      const resolver = new SpanResolver(content);
      
      expect(resolver.getText()).toBe("line1\nline2\nline3");
    });

    it('should handle mixed line endings', () => {
      const content = "line1\r\nline2\nline3\r\nline4";
      const resolver = new SpanResolver(content);
      
      expect(resolver.getText()).toBe("line1\nline2\nline3\nline4");
    });

    it('should preserve existing LF-only endings', () => {
      const content = "line1\nline2\nline3";
      const resolver = new SpanResolver(content);
      
      expect(resolver.getText()).toBe(content);
    });
  });

  describe('byte offset to line/column conversion', () => {
    const testContent = "hello\nworld\ntesting";
    // Byte positions: h(0) e(1) l(2) l(3) o(4) \n(5) w(6) o(7) r(8) l(9) d(10) \n(11) t(12) e(13) s(14) t(15) i(16) n(17) g(18)
    const resolver = new SpanResolver(testContent);

    it('should convert byte offset 0 to line 1, column 1', () => {
      const result = resolver.byteToLineCol(0);
      expect(result).toEqual({ line: 1, col: 1 });
    });

    it('should convert byte offset 5 (first newline) to line 1, column 6', () => {
      const result = resolver.byteToLineCol(5);
      expect(result).toEqual({ line: 1, col: 6 });
    });

    it('should convert byte offset 6 (start of second line) to line 2, column 1', () => {
      const result = resolver.byteToLineCol(6);
      expect(result).toEqual({ line: 2, col: 1 });
    });

    it('should convert byte offset 12 (start of third line) to line 3, column 1', () => {
      const result = resolver.byteToLineCol(12);
      expect(result).toEqual({ line: 3, col: 1 });
    });

    it('should convert byte offset 18 (end of file) to line 3, column 7', () => {
      const result = resolver.byteToLineCol(18);
      expect(result).toEqual({ line: 3, col: 7 });
    });

    it('should handle byte offset beyond file end', () => {
      const result = resolver.byteToLineCol(100);
      // Should clamp to end of file - "testing" has 7 characters, so position after it is col 8
      expect(result).toEqual({ line: 3, col: 8 }); // Should clamp to end of file
    });
  });

  describe('tab handling', () => {
    it('should treat tabs as single characters', () => {
      const content = "hello\tworld\ntest\ttabs";
      const resolver = new SpanResolver(content);
      
      // Tab at position 5 should be column 6
      const result1 = resolver.byteToLineCol(5);
      expect(result1).toEqual({ line: 1, col: 6 });
      
      // Character after tab should be column 7
      const result2 = resolver.byteToLineCol(6);
      expect(result2).toEqual({ line: 1, col: 7 });
    });

    it('should handle multiple tabs correctly', () => {
      const content = "\t\thello";
      const resolver = new SpanResolver(content);
      
      const result1 = resolver.byteToLineCol(0);
      expect(result1).toEqual({ line: 1, col: 1 });
      
      const result2 = resolver.byteToLineCol(1);
      expect(result2).toEqual({ line: 1, col: 2 });
      
      const result3 = resolver.byteToLineCol(2);
      expect(result3).toEqual({ line: 1, col: 3 });
    });
  });

  describe('Unicode handling', () => {
    it('should handle basic Unicode characters correctly', () => {
      const content = "héllo\nwørld";
      const resolver = new SpanResolver(content);
      
      // 'é' is 2 bytes in UTF-8, but should count as 1 code point
      const result1 = resolver.byteToLineCol(1);
      expect(result1).toEqual({ line: 1, col: 2 });
      
      // Test the newline position (which is at byte 6)
      const newlinePos = Buffer.from("héllo").length; // This is 6 (newline position)
      const result2 = resolver.byteToLineCol(newlinePos);
      expect(result2.line).toBe(2); // Newline is at the start of line 2
      expect(result2.col).toBe(1); // Column 1 of line 2
    });

    it('should handle emoji characters', () => {
      const content = "hello 🚀 world";
      const resolver = new SpanResolver(content);
      
      // Find the actual byte position of the emoji
      const emojiStart = Buffer.from("hello ").length; // Position after "hello "
      const result = resolver.byteToLineCol(emojiStart);
      expect(result.line).toBe(1);
      expect(result.col).toBe(7); // Should be at column 7 (after "hello ")
    });
  });

  describe('span resolution', () => {
    const testContent = `function test() {
  console.log("hello");
  return 42;
}`;
    const resolver = new SpanResolver(testContent);

    it('should resolve spans correctly', () => {
      // Test resolving the word "console" 
      const consoleStart = testContent.indexOf("console");
      const consoleEnd = consoleStart + "console".length;
      
      const result = resolver.resolveSpan(consoleStart, consoleEnd);
      
      expect(result.start.line).toBe(2);
      expect(result.start.col).toBe(3); // After 2 spaces of indentation
      expect(result.end.line).toBe(2);
      expect(result.end.col).toBe(10); // After "console"
    });

    it('should handle multi-line spans', () => {
      const start = testContent.indexOf("{");
      const end = testContent.lastIndexOf("}") + 1;
      
      const result = resolver.resolveSpan(start, end);
      
      expect(result.start.line).toBe(1);
      expect(result.start.col).toBe(17); // Position of opening brace
      expect(result.end.line).toBe(4);
      expect(result.end.col).toBe(2); // Position after closing brace
    });
  });

  describe('edge cases', () => {
    it('should handle empty content', () => {
      const resolver = new SpanResolver("");
      
      const result = resolver.byteToLineCol(0);
      expect(result).toEqual({ line: 1, col: 1 });
    });

    it('should handle single character content', () => {
      const resolver = new SpanResolver("x");
      
      const result1 = resolver.byteToLineCol(0);
      expect(result1).toEqual({ line: 1, col: 1 });
      
      const result2 = resolver.byteToLineCol(1);
      expect(result2).toEqual({ line: 1, col: 2 });
    });

    it('should handle content with only newlines', () => {
      const resolver = new SpanResolver("\n\n\n");
      
      const result1 = resolver.byteToLineCol(0);
      expect(result1).toEqual({ line: 1, col: 1 });
      
      const result2 = resolver.byteToLineCol(1);
      expect(result2).toEqual({ line: 2, col: 1 });
      
      const result3 = resolver.byteToLineCol(2);
      expect(result3).toEqual({ line: 3, col: 1 });
      
      const result4 = resolver.byteToLineCol(3);
      expect(result4).toEqual({ line: 4, col: 1 });
    });
  });
});