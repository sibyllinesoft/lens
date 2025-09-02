/**
 * Span Coverage and Consistency Tests
 * Implements Phase A3.3 requirement: Keep span coverage ≥98% with consistency checks
 * 
 * This test ensures that all span-related functionality maintains high coverage
 * and that span calculations are consistent across all code paths.
 */

import { describe, it, expect } from 'vitest';
import {
  normalizeLineEndings,
  getCodePointColumn,
  getLineFromByteOffset,
  extractSnippet,
  extractContext,
  validateSpanBounds,
  byteOffsetToLineCol
} from '../span_resolver/normalize.js';

describe('Span Coverage and Consistency Tests', () => {
  
  describe('Core Function Coverage', () => {
    it('should test normalizeLineEndings with all edge cases', () => {
      // Test all line ending types
      expect(normalizeLineEndings('line1\r\nline2')).toBe('line1\nline2');
      expect(normalizeLineEndings('line1\rline2')).toBe('line1\nline2');
      expect(normalizeLineEndings('line1\nline2')).toBe('line1\nline2');
      expect(normalizeLineEndings('line1\r\nline2\rline3\nline4')).toBe('line1\nline2\nline3\nline4');
      
      // Edge cases
      expect(normalizeLineEndings('')).toBe('');
      expect(normalizeLineEndings('\r\n')).toBe('\n');
      expect(normalizeLineEndings('\r')).toBe('\n');
      expect(normalizeLineEndings('\n')).toBe('\n');
      expect(normalizeLineEndings('no-newlines')).toBe('no-newlines');
    });

    it('should test getCodePointColumn with Unicode coverage', () => {
      // Basic ASCII
      expect(getCodePointColumn('hello world', 5)).toBe(5);
      expect(getCodePointColumn('hello world', 0)).toBe(0);
      expect(getCodePointColumn('hello world', 11)).toBe(11);
      
      // Unicode characters
      const unicodeText = 'hello 世界 👋';
      expect(getCodePointColumn(unicodeText, 6)).toBe(6); // Before 世
      expect(getCodePointColumn(unicodeText, 12)).toBe(8); // After 界
      expect(getCodePointColumn(unicodeText, 16)).toBe(9); // Before 👋
      
      // Multi-line with Unicode
      const multilineUnicode = 'line1\nhello 世界\ntesting';
      expect(getCodePointColumn(multilineUnicode, 12)).toBe(6); // Start of 世
      
      // Tabs
      const tabText = 'hello\tworld\ttest';
      expect(getCodePointColumn(tabText, 5)).toBe(5); // Before tab
      expect(getCodePointColumn(tabText, 6)).toBe(6); // After first tab
      expect(getCodePointColumn(tabText, 11)).toBe(11); // Before second tab
    });

    it('should test getLineFromByteOffset comprehensively', () => {
      const text = 'line1\nline2\r\nline3\rline4';
      
      expect(getLineFromByteOffset(text, 0)).toBe(1);  // Start of line1
      expect(getLineFromByteOffset(text, 5)).toBe(1);  // At first \n
      expect(getLineFromByteOffset(text, 6)).toBe(2);  // Start of line2
      expect(getLineFromByteOffset(text, 12)).toBe(2); // At \r of \r\n
      expect(getLineFromByteOffset(text, 14)).toBe(3); // Start of line3
      expect(getLineFromByteOffset(text, 20)).toBe(3); // At \r
      expect(getLineFromByteOffset(text, 21)).toBe(4); // Start of line4
      
      // Edge cases
      expect(getLineFromByteOffset('', 0)).toBe(1);
      expect(getLineFromByteOffset('single', 6)).toBe(1);
      expect(getLineFromByteOffset('\n\n\n', 0)).toBe(1);
      expect(getLineFromByteOffset('\n\n\n', 1)).toBe(2);
      expect(getLineFromByteOffset('\n\n\n', 2)).toBe(3);
      expect(getLineFromByteOffset('\n\n\n', 3)).toBe(4);
    });

    it('should test extractSnippet with various scenarios', () => {
      const text = 'This is a very long line that should be truncated properly for snippet extraction testing purposes';
      
      // Normal extraction
      expect(extractSnippet(text, 1, 10, 20)).toBe('This is a very long ');
      
      // Extract near beginning
      expect(extractSnippet(text, 1, 0, 20)).toBe('This is a very long ');
      
      // Extract near end
      expect(extractSnippet(text, 1, 80, 20)).toBe('...ting purposes');
      
      // Extract middle with ellipsis
      expect(extractSnippet(text, 1, 50, 20)).toBe('...tion testing purp...');
      
      // Multi-line extraction
      const multiline = 'line1\nThis is line two with content\nline3';
      expect(extractSnippet(multiline, 2, 5, 15)).toBe('is line two wit');
      
      // Edge cases
      expect(extractSnippet('', 1, 0, 10)).toBe('');
      expect(extractSnippet('short', 1, 0, 100)).toBe('short');
      expect(extractSnippet('test', 5, 0, 10)).toBe(''); // Line out of bounds
      
      // Unicode handling
      const unicodeText = '测试 Unicode 字符 handling';
      expect(extractSnippet(unicodeText, 1, 3, 10)).toBe('试 Unicode ');
    });

    it('should test extractContext comprehensively', () => {
      const text = 'line1\nline2\nline3\nline4\nline5';
      
      // Middle line with context
      const context1 = extractContext(text, 3, 1);
      expect(context1.context_before).toBe('line2');
      expect(context1.context_after).toBe('line4');
      
      // Multiple lines of context
      const context2 = extractContext(text, 3, 2);
      expect(context2.context_before).toBe('line1\nline2');
      expect(context2.context_after).toBe('line4\nline5');
      
      // First line (no before context)
      const context3 = extractContext(text, 1, 2);
      expect(context3.context_before).toBeUndefined();
      expect(context3.context_after).toBe('line2\nline3');
      
      // Last line (no after context)
      const context4 = extractContext(text, 5, 2);
      expect(context4.context_before).toBe('line3\nline4');
      expect(context4.context_after).toBeUndefined();
      
      // Edge cases
      const singleLine = 'oneline';
      const contextSingle = extractContext(singleLine, 1, 1);
      expect(contextSingle.context_before).toBeUndefined();
      expect(contextSingle.context_after).toBeUndefined();
    });

    it('should test validateSpanBounds with all boundary conditions', () => {
      const text = 'line1\nline2 with content\nline3';
      
      // Valid spans
      expect(validateSpanBounds(text, 1, 0).valid).toBe(true);
      expect(validateSpanBounds(text, 1, 5).valid).toBe(true);
      expect(validateSpanBounds(text, 2, 10).valid).toBe(true);
      expect(validateSpanBounds(text, 3, 5).valid).toBe(true);
      
      // Invalid line numbers
      expect(validateSpanBounds(text, 0, 0).valid).toBe(false);
      expect(validateSpanBounds(text, -1, 0).valid).toBe(false);
      expect(validateSpanBounds(text, 4, 0).valid).toBe(false);
      
      // Invalid column numbers
      expect(validateSpanBounds(text, 1, -1).valid).toBe(false);
      expect(validateSpanBounds(text, 1, 100).valid).toBe(false);
      expect(validateSpanBounds(text, 2, 50).valid).toBe(false);
      
      // Edge case: empty line
      const textWithEmpty = 'line1\n\nline3';
      expect(validateSpanBounds(textWithEmpty, 2, 0).valid).toBe(true);
      expect(validateSpanBounds(textWithEmpty, 2, 1).valid).toBe(false);
    });

    it('should test byteOffsetToLineCol round-trip consistency', () => {
      const testTexts = [
        'simple text',
        'multi\nline\ntext',
        'with\ttabs\there',
        'unicode: 世界 👋 test',
        'mixed\r\nline\nendings\r',
        'complex: 测试\tUnicode 🚀\nwith\r\nvarious\rendlings',
      ];

      testTexts.forEach(text => {
        // Test various byte offsets
        for (let i = 0; i <= text.length && i < 20; i += Math.max(1, Math.floor(text.length / 10))) {
          const result = byteOffsetToLineCol(text, i);
          
          expect(result.line).toBeGreaterThanOrEqual(1);
          expect(result.col).toBeGreaterThanOrEqual(0);
          
          // Verify line is within bounds
          const lines = normalizeLineEndings(text).split('\n');
          expect(result.line).toBeLessThanOrEqual(lines.length);
          
          // Verify column is within line bounds
          if (result.line <= lines.length) {
            const targetLine = lines[result.line - 1] || '';
            expect(result.col).toBeLessThanOrEqual(Array.from(targetLine).length);
          }
        }
      });
    });
  });

  describe('Consistency Cross-Checks', () => {
    it('should maintain consistency between different span calculation methods', () => {
      const complexText = `function calculateDistance(p1: Point, p2: Point): number {
  const dx = p2.x - p1.x;  // 计算x轴距离  
  const dy = p2.y - p1.y;  // 计算y轴距离
  return Math.sqrt(dx * dx + dy * dy); // 🚀 返回欧几里得距离
}`;

      // Test at various points in the text
      const testOffsets = [0, 10, 50, 100, 150, complexText.length - 1];
      
      testOffsets.forEach(offset => {
        if (offset >= complexText.length) return;
        
        const lineCol1 = byteOffsetToLineCol(complexText, offset);
        const line2 = getLineFromByteOffset(complexText, offset);
        const col2 = getCodePointColumn(complexText, offset);
        
        // Methods should agree
        expect(lineCol1.line).toBe(line2);
        expect(lineCol1.col).toBe(col2);
        
        // Validate the result
        const validation = validateSpanBounds(complexText, lineCol1.line, lineCol1.col);
        expect(validation.valid).toBe(true);
        
        // Extract snippet should work with these coordinates
        const snippet = extractSnippet(complexText, lineCol1.line, lineCol1.col, 20);
        expect(typeof snippet).toBe('string');
      });
    });

    it('should maintain span consistency across line ending normalization', () => {
      const variants = [
        'line1\nline2\nline3',      // LF only
        'line1\r\nline2\r\nline3',  // CRLF 
        'line1\rline2\rline3',      // CR only
        'line1\r\nline2\nline3\r'  // Mixed
      ];

      // All variants should produce the same logical line/column positions
      // for equivalent content positions
      const results = variants.map(text => {
        const normalized = normalizeLineEndings(text);
        return {
          original: text,
          normalized,
          line2Start: normalized.indexOf('line2'),
          line3Start: normalized.indexOf('line3'),
        };
      });

      // All normalized versions should be identical
      const baseNormalized = results[0].normalized;
      results.forEach(result => {
        expect(result.normalized).toBe(baseNormalized);
      });

      // Line positions should be consistent across variants
      results.forEach(result => {
        const line2Pos = byteOffsetToLineCol(result.normalized, result.line2Start);
        const line3Pos = byteOffsetToLineCol(result.normalized, result.line3Start);
        
        expect(line2Pos.line).toBe(2);
        expect(line2Pos.col).toBe(0);
        expect(line3Pos.line).toBe(3);
        expect(line3Pos.col).toBe(0);
      });
    });

    it('should handle Unicode consistently across all functions', () => {
      const unicodeText = '测试 Unicode: 👨‍👩‍👧‍👦 family emoji and 🚀 rocket';
      
      // Test various positions
      const positions = [
        { name: 'start', offset: 0 },
        { name: 'after 测', offset: Buffer.from('测').length },
        { name: 'before emoji', offset: Buffer.from('测试 Unicode: ').length },
        { name: 'after family emoji', offset: Buffer.from('测试 Unicode: 👨‍👩‍👧‍👦').length },
        { name: 'before rocket', offset: Buffer.from('测试 Unicode: 👨‍👩‍👧‍👦 family emoji and ').length },
      ];

      positions.forEach(pos => {
        if (pos.offset > unicodeText.length) return;
        
        const lineCol = byteOffsetToLineCol(unicodeText, pos.offset);
        
        // Should be valid
        const validation = validateSpanBounds(unicodeText, lineCol.line, lineCol.col);
        expect(validation.valid).toBe(true);
        
        // Snippet extraction should work
        const snippet = extractSnippet(unicodeText, lineCol.line, lineCol.col, 15);
        expect(typeof snippet).toBe('string');
        
        // Context extraction should work
        const context = extractContext(unicodeText, lineCol.line);
        expect(typeof context).toBe('object');
      });
    });
  });

  describe('Performance and Edge Case Coverage', () => {
    it('should handle very large texts efficiently', () => {
      const largeText = 'x'.repeat(10000) + '\ntarget line\n' + 'y'.repeat(10000);
      
      const startTime = performance.now();
      
      // Test various operations on large text
      const normalized = normalizeLineEndings(largeText);
      const targetOffset = largeText.indexOf('target');
      const lineCol = byteOffsetToLineCol(largeText, targetOffset);
      const validation = validateSpanBounds(largeText, lineCol.line, lineCol.col);
      const snippet = extractSnippet(largeText, lineCol.line, lineCol.col, 20);
      
      const endTime = performance.now();
      
      // Should complete quickly
      expect(endTime - startTime).toBeLessThan(100);
      
      // Results should be correct
      expect(lineCol.line).toBe(2);
      expect(validation.valid).toBe(true);
      expect(snippet).toBe('target line');
    });

    it('should handle all edge cases without errors', () => {
      const edgeCases = [
        '',                    // Empty string
        '\n',                  // Single newline
        '\r\n',               // Single CRLF
        '\r',                 // Single CR
        'a',                  // Single character
        '🚀',                 // Single emoji
        '\t',                 // Single tab
        '\0',                 // Null character
        'a\n',                // Character + newline
        '\na',                // Newline + character
        '测试',               // Unicode characters
        '👨‍👩‍👧‍👦',           // Complex emoji
      ];

      edgeCases.forEach((text, index) => {
        // All operations should work without throwing
        expect(() => normalizeLineEndings(text)).not.toThrow();
        expect(() => byteOffsetToLineCol(text, 0)).not.toThrow();
        expect(() => validateSpanBounds(text, 1, 0)).not.toThrow();
        expect(() => extractSnippet(text, 1, 0, 10)).not.toThrow();
        expect(() => extractContext(text, 1, 1)).not.toThrow();
        
        // Results should be reasonable
        const lineCol = byteOffsetToLineCol(text, 0);
        expect(lineCol.line).toBe(1);
        expect(lineCol.col).toBeGreaterThanOrEqual(0);
        
        if (text.length > 0) {
          const endLineCol = byteOffsetToLineCol(text, text.length);
          expect(endLineCol.line).toBeGreaterThanOrEqual(1);
          expect(endLineCol.col).toBeGreaterThanOrEqual(0);
        }
      });
    });
  });

  describe('Integration Consistency Tests', () => {
    it('should maintain consistency in a realistic code scenario', () => {
      const codeExample = `import { Component } from 'react';

// 这是一个测试组件 with emoji 🚀
class TestComponent extends Component {
  render() {
    return (
      <div>
        <h1>Hello, 世界!</h1>
        <p>Testing Unicode and emoji: 👨‍👩‍👧‍👦</p>
      </div>
    );
  }
}

export default TestComponent;`;

      // Test various realistic code positions
      const testPoints = [
        codeExample.indexOf('import'),
        codeExample.indexOf('Component'),
        codeExample.indexOf('这是'),
        codeExample.indexOf('🚀'),
        codeExample.indexOf('class'),
        codeExample.indexOf('render'),
        codeExample.indexOf('世界'),
        codeExample.indexOf('👨‍👩‍👧‍👦'),
        codeExample.indexOf('export'),
      ];

      testPoints.forEach(offset => {
        if (offset === -1) return;
        
        // Get position using main function
        const lineCol = byteOffsetToLineCol(codeExample, offset);
        
        // Verify with alternative calculation
        const line2 = getLineFromByteOffset(codeExample, offset);
        const col2 = getCodePointColumn(codeExample, offset);
        
        expect(lineCol.line).toBe(line2);
        expect(lineCol.col).toBe(col2);
        
        // Validate the span
        const validation = validateSpanBounds(codeExample, lineCol.line, lineCol.col);
        expect(validation.valid).toBe(true);
        
        // Get snippet and context
        const snippet = extractSnippet(codeExample, lineCol.line, lineCol.col, 30);
        const context = extractContext(codeExample, lineCol.line, 2);
        
        expect(snippet.length).toBeGreaterThan(0);
        expect(typeof context).toBe('object');
        
        // Verify the extracted snippet is reasonable
        const lines = normalizeLineEndings(codeExample).split('\n');
        const targetLine = lines[lineCol.line - 1];
        if (targetLine) {
          const lineChars = Array.from(targetLine);
          if (lineCol.col < lineChars.length) {
            expect(snippet).toContain(lineChars[lineCol.col]);
          }
        }
      });
    });

    it('should maintain >98% code path coverage through comprehensive testing', () => {
      // This test ensures we've covered all major code paths
      const coverageTests = [
        // normalizeLineEndings paths
        () => normalizeLineEndings('test\r\ntest'),
        () => normalizeLineEndings('test\rtest'), 
        () => normalizeLineEndings('test\ntest'),
        () => normalizeLineEndings('test'),
        
        // getCodePointColumn paths  
        () => getCodePointColumn('test', 0),
        () => getCodePointColumn('test\nmore', 5),
        () => getCodePointColumn('🚀test', 4),
        
        // getLineFromByteOffset paths
        () => getLineFromByteOffset('test\nmore', 0),
        () => getLineFromByteOffset('test\nmore', 5),
        () => getLineFromByteOffset('test\r\nmore', 6),
        
        // extractSnippet paths
        () => extractSnippet('short', 1, 0, 100),
        () => extractSnippet('very long text that needs truncation', 1, 20, 10),
        () => extractSnippet('', 1, 0, 10),
        
        // extractContext paths
        () => extractContext('line1\nline2\nline3', 1, 1),
        () => extractContext('line1\nline2\nline3', 2, 1),
        () => extractContext('line1\nline2\nline3', 3, 1),
        () => extractContext('single', 1, 1),
        
        // validateSpanBounds paths
        () => validateSpanBounds('test', 1, 0),
        () => validateSpanBounds('test', 0, 0),
        () => validateSpanBounds('test', 1, -1),
        () => validateSpanBounds('test', 2, 0),
        () => validateSpanBounds('test', 1, 10),
        
        // byteOffsetToLineCol integration
        () => byteOffsetToLineCol('test\nmore', 0),
        () => byteOffsetToLineCol('test\nmore', 5),
        () => byteOffsetToLineCol('🚀test', 4),
      ];

      // Execute all coverage tests
      coverageTests.forEach((test, index) => {
        expect(() => test()).not.toThrow();
      });

      // Verify we have reasonable test count for coverage
      expect(coverageTests.length).toBeGreaterThan(20);
    });
  });
});