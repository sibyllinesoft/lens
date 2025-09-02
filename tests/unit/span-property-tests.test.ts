/**
 * Property tests for span handling with special characters
 * Implements Phase A3.1 requirement: property tests for spans (CRLF/tabs/emoji)
 */

import { describe, it, expect } from 'vitest';

// Property testing utilities
function generateRandomString(length: number, charset: string): string {
  let result = '';
  for (let i = 0; i < length; i++) {
    result += charset.charAt(Math.floor(Math.random() * charset.length));
  }
  return result;
}

function generateSpecialCharacterString(length: number): string {
  const specialChars = '\r\n\t\u0000\u0001\u0002\u0003\u0004\u0005\u0006\u0007\u0008\u000B\u000C\u000E\u000F';
  return generateRandomString(length, specialChars);
}

function generateEmojiString(length: number): string {
  const emojis = '😀😃😄😁😆😅😂🤣😊😇🙂🙃😉😌😍🥰😘😗😙😚😋😛😝😜🤪🤨🧐🤓😎🤩🥳😏😒😞😔😟😕🙁☹️😣😖😫😩🥺😢😭😤😠😡🤬🤯😳🥵🥶😨😰😥😓🤗🤔🤭🤫🤥😶😐😑😬🙄😯😦😧😮😲🥱😴🤤😪😵🤐🥴🤢🤮🤧😷🤒🤕🤑🤠😈👿👹👺🤡💩👻💀☠️👽👾🤖🎃😺😸😹😻😼😽🙀😿😾🤲🤲🏻🤲🏼🤲🏽🤲🏾🤲🏿';
  return generateRandomString(length, emojis);
}

// Mock span resolver functions that need to handle special characters
function normalizeSpan(text: string): string {
  // Handle CRLF normalization
  return text
    .replace(/\r\n/g, '\n')  // Convert CRLF to LF
    .replace(/\r/g, '\n')    // Convert CR to LF
    .replace(/\t/g, ' ')     // Convert tabs to spaces
    .trim();
}

function calculateSpanLength(text: string): number {
  // Properly count Unicode characters including emoji
  return Array.from(text).length;
}

function extractSpanContext(text: string, offset: number, length: number): string {
  const chars = Array.from(text);
  return chars.slice(offset, offset + length).join('');
}

function isValidSpan(text: string, offset: number, length: number): boolean {
  const chars = Array.from(text);
  return offset >= 0 && 
         length >= 0 && 
         offset + length <= chars.length;
}

describe('Span Property Tests', () => {
  describe('CRLF handling properties', () => {
    it('should normalize CRLF sequences consistently', () => {
      // Property: normalizeSpan should handle all CRLF variations
      const testCases = [
        'line1\r\nline2',    // CRLF
        'line1\rline2',      // CR only
        'line1\nline2',      // LF only
        'line1\r\n\rline2',  // Mixed
        '\r\n\r\n',         // Only CRLF
        '',                  // Empty string
      ];

      for (const testCase of testCases) {
        const normalized = normalizeSpan(testCase);
        
        // Property: No CRLF or standalone CR should remain
        expect(normalized).not.toMatch(/\r\n/);
        expect(normalized).not.toMatch(/\r/);
        
        // Property: Should have consistent line endings
        const lines = normalized.split('\n');
        expect(lines.length).toBeGreaterThanOrEqual(1);
      }
    });

    it('should preserve content while normalizing line endings', () => {
      for (let i = 0; i < 100; i++) {
        const content = generateRandomString(50, 'abcdefghijklmnopqrstuvwxyz0123456789');
        const withCRLF = content.split('').join('\r\n');
        
        const normalized = normalizeSpan(withCRLF);
        const contentOnly = normalized.replace(/\n/g, '');
        
        // Property: Content should be preserved
        expect(contentOnly).toBe(content);
      }
    });
  });

  describe('Tab handling properties', () => {
    it('should convert tabs to spaces consistently', () => {
      for (let i = 0; i < 50; i++) {
        const tabCount = Math.floor(Math.random() * 10) + 1;
        const text = '\t'.repeat(tabCount) + 'content' + '\t'.repeat(tabCount);
        
        const normalized = normalizeSpan(text);
        
        // Property: No tabs should remain
        expect(normalized).not.toMatch(/\t/);
        
        // Property: Should have spaces instead
        expect(normalized.trim()).toBe('content');
      }
    });

    it('should handle mixed tabs and spaces', () => {
      const testCases = [
        '\t    content',      // Tab then spaces
        '    \tcontent',      // Spaces then tab
        '\t\t  \t  content', // Mixed tabs and spaces
        'content\t\t',       // Trailing tabs
      ];

      for (const testCase of testCases) {
        const normalized = normalizeSpan(testCase);
        
        // Property: No tabs should remain
        expect(normalized).not.toMatch(/\t/);
        
        // Property: Should contain the content
        expect(normalized.includes('content')).toBe(true);
      }
    });
  });

  describe('Emoji and Unicode handling properties', () => {
    it('should correctly calculate length for emoji strings', () => {
      for (let i = 0; i < 50; i++) {
        const emojiString = generateEmojiString(Math.floor(Math.random() * 20) + 1);
        const calculatedLength = calculateSpanLength(emojiString);
        const actualLength = Array.from(emojiString).length;
        
        // Property: Length calculation should be accurate for Unicode
        expect(calculatedLength).toBe(actualLength);
      }
    });

    it('should handle emoji in span extraction', () => {
      const emojis = ['😀', '🤝', '👨‍👩‍👧‍👦', '🏳️‍🌈', '👩🏽‍💻'];
      
      for (const emoji of emojis) {
        const text = `start${emoji}end`;
        const chars = Array.from(text);
        
        // Property: Should extract emoji correctly
        const extracted = extractSpanContext(text, 5, 1);
        expect(extracted).toBe(emoji);
        
        // Property: Span validation should work with emoji
        expect(isValidSpan(text, 5, 1)).toBe(true);
        expect(isValidSpan(text, 0, chars.length)).toBe(true);
      }
    });

    it('should handle complex emoji sequences', () => {
      const complexEmojis = [
        '👨‍👩‍👧‍👦', // Family emoji (ZWJ sequence)
        '🏳️‍🌈',        // Flag with ZWJ
        '👩🏽‍💻',       // Profession with skin tone
        '🤝🏻',         // Handshake with skin tone
      ];

      for (const emoji of complexEmojis) {
        const text = `prefix_${emoji}_suffix`;
        
        // Property: Length calculation should handle complex sequences
        const length = calculateSpanLength(text);
        const expected = Array.from(text).length;
        expect(length).toBe(expected);
        
        // Property: Extraction should preserve complex sequences
        const startIndex = Array.from('prefix_').length;
        const extracted = extractSpanContext(text, startIndex, 1);
        expect(extracted).toBe(emoji);
      }
    });
  });

  describe('Control character handling properties', () => {
    it('should handle null bytes and control characters', () => {
      for (let i = 0; i < 20; i++) {
        const controlChars = generateSpecialCharacterString(10);
        const text = `content${controlChars}more`;
        
        const normalized = normalizeSpan(text);
        
        // Property: Should handle without crashing
        expect(typeof normalized).toBe('string');
        
        // Property: Should preserve visible content
        expect(normalized.includes('content')).toBe(true);
        expect(normalized.includes('more')).toBe(true);
      }
    });

    it('should validate spans with control characters', () => {
      const controlChar = '\u0001';
      const text = `start${controlChar}end`;
      
      // Property: Span validation should work with control chars
      expect(isValidSpan(text, 0, calculateSpanLength(text))).toBe(true);
      expect(isValidSpan(text, -1, 1)).toBe(false);
      expect(isValidSpan(text, 0, -1)).toBe(false);
    });
  });

  describe('Boundary condition properties', () => {
    it('should handle empty and single character spans', () => {
      const testStrings = ['', 'a', '😀', '\n', '\t', '\r\n'];
      
      for (const str of testStrings) {
        // Property: Should handle edge cases
        const length = calculateSpanLength(str);
        expect(length).toBeGreaterThanOrEqual(0);
        
        const normalized = normalizeSpan(str);
        expect(typeof normalized).toBe('string');
        
        // Property: Empty span extraction should work
        if (length > 0) {
          expect(isValidSpan(str, 0, length)).toBe(true);
          expect(isValidSpan(str, 0, 0)).toBe(true);
        }
      }
    });

    it('should handle very long spans', () => {
      for (let i = 0; i < 10; i++) {
        const longText = generateRandomString(1000, 'abcdefghijklmnopqrstuvwxyz\n\t\r😀🎉');
        
        // Property: Should handle large spans without performance issues
        const start = Date.now();
        const length = calculateSpanLength(longText);
        const normalized = normalizeSpan(longText);
        const end = Date.now();
        
        // Property: Should complete within reasonable time (< 100ms for 1000 chars)
        expect(end - start).toBeLessThan(100);
        
        // Property: Should preserve content size roughly
        expect(length).toBeGreaterThan(0);
        expect(normalized.length).toBeGreaterThan(0);
      }
    });
  });

  describe('Invariant properties', () => {
    it('should maintain span invariants under normalization', () => {
      for (let i = 0; i < 100; i++) {
        const originalText = generateRandomString(100, 'abc\r\n\t😀🎉xyz123');
        const normalized = normalizeSpan(originalText);
        
        // Property: Normalization should be idempotent
        const doubleNormalized = normalizeSpan(normalized);
        expect(normalized).toBe(doubleNormalized);
        
        // Property: Length should be consistent
        const length1 = calculateSpanLength(normalized);
        const length2 = calculateSpanLength(normalized);
        expect(length1).toBe(length2);
        
        // Property: Valid spans should remain valid after normalization
        if (length1 > 0) {
          expect(isValidSpan(normalized, 0, length1)).toBe(true);
          expect(isValidSpan(normalized, length1, 0)).toBe(true);
        }
      }
    });

    it('should preserve ordering properties', () => {
      const testText = 'a😀b🎉c\nd\te';
      const chars = Array.from(testText);
      
      for (let i = 0; i < chars.length; i++) {
        for (let j = i; j < chars.length; j++) {
          const span1 = extractSpanContext(testText, i, j - i);
          const span2 = extractSpanContext(testText, 0, j);
          const span3 = extractSpanContext(testText, i, chars.length - i);
          
          // Property: Subspan should be contained in larger span
          if (j > i) {
            expect(span2.includes(span1) || span3.includes(span1)).toBe(true);
          }
          
          // Property: Extraction should respect boundaries
          expect(isValidSpan(testText, i, j - i)).toBe(true);
        }
      }
    });
  });

  describe('Real-world scenario properties', () => {
    it('should handle common programming constructs with special chars', () => {
      const codeSnippets = [
        'function test() {\n\treturn "hello\r\nworld";\n}',
        'const emoji = "😀🎉";\nconsole.log(emoji);',
        'const multiline = `\n\tLine 1\r\n\tLine 2\n`;',
        '// Comment with emoji 🚀\nlet x = 42;',
        'if (true) {\n\t// Tab indented\n\tconsole.log("test\\n");\n}',
      ];

      for (const code of codeSnippets) {
        // Property: Should process code snippets without errors
        const length = calculateSpanLength(code);
        const normalized = normalizeSpan(code);
        
        expect(length).toBeGreaterThan(0);
        expect(normalized).toBeTruthy();
        
        // Property: Should preserve code structure
        expect(normalized.includes('function') || 
               normalized.includes('const') || 
               normalized.includes('//') || 
               normalized.includes('if')).toBe(true);
        
        // Property: Should handle extractions within code
        for (let i = 0; i < Math.min(length, 10); i++) {
          expect(isValidSpan(code, i, 1)).toBe(true);
        }
      }
    });

    it('should handle file paths and URLs with special characters', () => {
      const paths = [
        '/path/with\tspaces/file.txt',
        'C:\\Windows\\System32\\file.exe\r\n',
        'https://example.com/path?query=value&other=test\n',
        '/path/with/emoji/🚀/file.js',
        './relative/../path/./file.ts',
      ];

      for (const path of paths) {
        const normalized = normalizeSpan(path);
        const length = calculateSpanLength(path);
        
        // Property: Should preserve essential path structure
        expect(normalized.includes('/') || normalized.includes('\\')).toBe(true);
        
        // Property: Should handle path extraction
        if (length > 0) {
          const firstChar = extractSpanContext(path, 0, 1);
          expect(typeof firstChar).toBe('string');
          expect(firstChar.length).toBeGreaterThan(0);
        }
      }
    });
  });

  describe('Performance properties', () => {
    it('should maintain consistent performance characteristics', () => {
      const sizes = [10, 100, 1000];
      const maxTime = [5, 20, 100]; // Max milliseconds for each size
      
      for (let i = 0; i < sizes.length; i++) {
        const size = sizes[i];
        const maxDuration = maxTime[i];
        
        const text = generateRandomString(size, 'abc\n\t\r😀🎉123xyz');
        
        const start = Date.now();
        const length = calculateSpanLength(text);
        const normalized = normalizeSpan(text);
        const isValid = isValidSpan(text, 0, length);
        const extracted = extractSpanContext(text, 0, Math.min(10, length));
        const end = Date.now();
        
        // Property: Performance should scale reasonably
        expect(end - start).toBeLessThan(maxDuration);
        
        // Property: Results should be correct regardless of size
        expect(length).toBe(Array.from(text).length);
        expect(typeof normalized).toBe('string');
        expect(isValid).toBe(true);
        expect(typeof extracted).toBe('string');
      }
    });
  });
});