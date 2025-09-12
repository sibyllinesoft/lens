import { describe, it, expect, beforeEach } from 'bun:test';
import { StageAAdapter, StageBAdapter, StageCAdapter } from './adapters';

describe('SpanResolver Adapters', () => {
  const sampleContent = `function hello(name) {
  console.log("Hello, " + name);
  return name.toUpperCase();
}`;

  describe('StageAAdapter', () => {
    let adapter: StageAAdapter;

    beforeEach(() => {
      adapter = new StageAAdapter();
    });

    it('should create resolver with provided content', () => {
      const resolver = adapter.createResolver(sampleContent);
      
      expect(resolver.getText()).toBe(sampleContent);
    });

    it('should resolve spans correctly', () => {
      const resolver = adapter.createResolver(sampleContent);
      
      // Find "console" and resolve its span
      const consolePos = sampleContent.indexOf("console");
      const result = resolver.resolveSpan(consolePos, consolePos + 7);
      
      expect(result.start.line).toBe(2);
      expect(result.start.col).toBe(3);
      expect(result.end.line).toBe(2);
      expect(result.end.col).toBe(10);
    });

    it('should handle empty content gracefully', () => {
      const resolver = adapter.createResolver("");
      const result = resolver.byteToLineCol(0);
      
      expect(result).toEqual({ line: 1, col: 1 });
    });
  });

  describe('StageBAdapter', () => {
    let adapter: StageBAdapter;

    beforeEach(() => {
      adapter = new StageBAdapter();
    });

    it('should normalize line endings during resolution', () => {
      const crlfContent = "line1\r\nline2\r\nconsole.log();";
      const resolver = adapter.createResolver(crlfContent);
      
      // Should normalize CRLF to LF
      expect(resolver.getText()).toBe("line1\nline2\nconsole.log();");
    });

    it('should resolve spans in normalized content', () => {
      const crlfContent = "function test() {\r\n  console.log();\r\n}";
      const resolver = adapter.createResolver(crlfContent);
      
      const consolePos = resolver.getText().indexOf("console");
      const result = resolver.resolveSpan(consolePos, consolePos + 7);
      
      expect(result.start.line).toBe(2);
      expect(result.start.col).toBe(3);
    });

    it('should handle mixed line endings', () => {
      const mixedContent = "line1\r\nline2\nline3\r\nend";
      const resolver = adapter.createResolver(mixedContent);
      
      expect(resolver.getText()).toBe("line1\nline2\nline3\nend");
    });
  });

  describe('StageCAdapter', () => {
    let adapter: StageCAdapter;

    beforeEach(() => {
      adapter = new StageCAdapter();
    });

    it('should handle Unicode characters correctly', () => {
      const unicodeContent = "function café() {\n  return '🚀';\n}";
      const resolver = adapter.createResolver(unicodeContent);
      
      // Find the emoji position
      const emojiPos = resolver.getText().indexOf("🚀");
      const result = resolver.byteToLineCol(emojiPos);
      
      expect(result.line).toBe(2);
      expect(result.col).toBeGreaterThan(1); // Should be positioned correctly
    });

    it('should handle tab characters as single positions', () => {
      const tabContent = "function test() {\n\tconsole.log('tab test');\n}";
      const resolver = adapter.createResolver(tabContent);
      
      const consolePos = resolver.getText().indexOf("console");
      const result = resolver.byteToLineCol(consolePos);
      
      expect(result.line).toBe(2);
      expect(result.col).toBe(2); // Tab counts as 1 character
    });

    it('should resolve complex Unicode spans', () => {
      const complexContent = "const msg = 'héllo 🌟 wørld';\nconsole.log(msg);";
      const resolver = adapter.createResolver(complexContent);
      
      const consolePos = resolver.getText().indexOf("console");
      const result = resolver.resolveSpan(consolePos, consolePos + "console.log".length);
      
      expect(result.start.line).toBe(2);
      expect(result.start.col).toBe(1);
      expect(result.end.line).toBe(2);
      expect(result.end.col).toBe(12);
    });
  });

  describe('Adapter Integration', () => {
    const testContent = `import { test } from 'lib';

function processData(data) {
  // This is a comment
  const result = data.map(x => x * 2);
  return result.filter(x => x > 0);
}

export default processData;`;

    it('should produce consistent results across all adapters for simple spans', () => {
      const stageA = new StageAAdapter();
      const stageB = new StageBAdapter();
      const stageC = new StageCAdapter();

      const resolverA = stageA.createResolver(testContent);
      const resolverB = stageB.createResolver(testContent);
      const resolverC = stageC.createResolver(testContent);

      // Find "processData" function name
      const funcPos = testContent.indexOf("processData");
      const funcEnd = funcPos + "processData".length;

      const resultA = resolverA.resolveSpan(funcPos, funcEnd);
      const resultB = resolverB.resolveSpan(funcPos, funcEnd);
      const resultC = resolverC.resolveSpan(funcPos, funcEnd);

      // All adapters should produce the same result for ASCII content
      expect(resultA).toEqual(resultB);
      expect(resultB).toEqual(resultC);
    });

    it('should handle different content types appropriately', () => {
      const stageA = new StageAAdapter();
      const stageB = new StageBAdapter();
      const stageC = new StageCAdapter();

      // Test with CRLF content
      const crlfContent = "line1\r\nline2\r\ntest";
      const resolverA = stageA.createResolver(crlfContent);
      const resolverB = stageB.createResolver(crlfContent);
      const resolverC = stageC.createResolver(crlfContent);

      // Stage A preserves original
      expect(resolverA.getText()).toBe(crlfContent);
      
      // Stages B and C normalize
      expect(resolverB.getText()).toBe("line1\nline2\ntest");
      expect(resolverC.getText()).toBe("line1\nline2\ntest");
    });
  });

  describe('Performance and Edge Cases', () => {
    it('should handle large files efficiently', () => {
      const largeContent = "function test() {\n".repeat(1000) + "}".repeat(1000);
      const adapter = new StageCAdapter();
      const resolver = adapter.createResolver(largeContent);

      const start = performance.now();
      resolver.byteToLineCol(largeContent.length - 1);
      const end = performance.now();

      // Should complete within reasonable time (< 100ms for large file)
      expect(end - start).toBeLessThan(100);
    });

    it('should handle deeply nested content', () => {
      const nestedContent = "{".repeat(100) + "test" + "}".repeat(100);
      const adapter = new StageBAdapter();
      const resolver = adapter.createResolver(nestedContent);

      const testPos = nestedContent.indexOf("test");
      const result = resolver.byteToLineCol(testPos);

      expect(result.line).toBe(1);
      expect(result.col).toBe(101); // After 100 opening braces
    });
  });
});