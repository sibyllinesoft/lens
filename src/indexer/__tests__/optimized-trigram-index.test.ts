/**
 * Phase B1 Optimization Tests: Roaring Bitmap-based Trigram Index
 * Comprehensive test suite for bitmap operations and performance validation
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { OptimizedTrigramIndex } from '../optimized-trigram-index.js';
import type { DocumentPosition } from '../../types/core.js';

describe('OptimizedTrigramIndex', () => {
  let index: OptimizedTrigramIndex;

  beforeEach(() => {
    index = new OptimizedTrigramIndex();
  });

  describe('Document Management', () => {
    it('should add documents with trigrams and positions', () => {
      const positions: DocumentPosition[] = [
        {
          doc_id: 'doc1',
          file_path: '/test/file1.ts',
          line: 1,
          col: 0,
          length: 5,
        },
      ];

      index.addDocument('doc1', ['abc', 'bcd', 'cde'], positions);

      const stats = index.getStats();
      expect(stats.document_count).toBe(1);
      expect(stats.trigram_count).toBe(3);
    });

    it('should handle multiple documents with overlapping trigrams', () => {
      const positions1: DocumentPosition[] = [
        { doc_id: 'doc1', file_path: '/test/file1.ts', line: 1, col: 0, length: 5 },
      ];
      const positions2: DocumentPosition[] = [
        { doc_id: 'doc2', file_path: '/test/file2.ts', line: 1, col: 0, length: 5 },
      ];

      index.addDocument('doc1', ['abc', 'bcd'], positions1);
      index.addDocument('doc2', ['bcd', 'cde'], positions2);

      // Should have 3 unique trigrams across 2 documents
      const stats = index.getStats();
      expect(stats.document_count).toBe(2);
      expect(stats.trigram_count).toBe(3);
    });

    it('should assign sequential numeric document indices', () => {
      const positions: DocumentPosition[] = [
        { doc_id: 'doc1', file_path: '/test/file1.ts', line: 1, col: 0, length: 5 },
      ];

      index.addDocument('doc1', ['abc'], positions);
      const docIndex = index.getDocumentIndex('doc1');
      expect(docIndex).toBe(0);

      index.addDocument('doc2', ['def'], positions);
      const docIndex2 = index.getDocumentIndex('doc2');
      expect(docIndex2).toBe(1);
    });

    it('should retrieve document IDs from indices', () => {
      const positions: DocumentPosition[] = [
        { doc_id: 'doc1', file_path: '/test/file1.ts', line: 1, col: 0, length: 5 },
      ];

      index.addDocument('test-doc', ['abc'], positions);
      const docIndex = index.getDocumentIndex('test-doc');
      expect(docIndex).toBeDefined();
      
      const retrievedDocId = index.getDocumentId(docIndex!);
      expect(retrievedDocId).toBe('test-doc');
    });
  });

  describe('Bitmap Trigram Intersection', () => {
    beforeEach(() => {
      // Set up test data with overlapping trigrams
      const positions1: DocumentPosition[] = [
        { doc_id: 'doc1', file_path: '/test/file1.ts', line: 1, col: 0, length: 8 },
      ];
      const positions2: DocumentPosition[] = [
        { doc_id: 'doc2', file_path: '/test/file2.ts', line: 2, col: 0, length: 8 },
      ];
      const positions3: DocumentPosition[] = [
        { doc_id: 'doc3', file_path: '/test/file3.ts', line: 3, col: 0, length: 8 },
      ];

      // doc1 has trigrams: abc, bcd, cde
      index.addDocument('doc1', ['abc', 'bcd', 'cde'], positions1);
      
      // doc2 has trigrams: bcd, cde, def (shares 'bcd', 'cde' with doc1)
      index.addDocument('doc2', ['bcd', 'cde', 'def'], positions2);
      
      // doc3 has trigrams: cde, def, efg (shares 'cde' with doc1 and doc2, 'def' with doc2)
      index.addDocument('doc3', ['cde', 'def', 'efg'], positions3);
    });

    it('should find documents containing all specified trigrams', () => {
      // Find documents containing both 'bcd' and 'cde'
      const docIndices = index.findDocumentsContainingAllTrigrams(['bcd', 'cde']);
      
      // Should return doc1 and doc2 (both contain 'bcd' and 'cde')
      expect(docIndices).toHaveLength(2);
      
      const docIds = docIndices.map(idx => index.getDocumentId(idx));
      expect(docIds).toContain('doc1');
      expect(docIds).toContain('doc2');
      expect(docIds).not.toContain('doc3'); // doc3 doesn't have 'bcd'
    });

    it('should return empty array for non-existent trigrams', () => {
      const docIndices = index.findDocumentsContainingAllTrigrams(['xyz']);
      expect(docIndices).toHaveLength(0);
    });

    it('should handle single trigram searches', () => {
      const docIndices = index.findDocumentsContainingAllTrigrams(['cde']);
      
      // All three documents contain 'cde'
      expect(docIndices).toHaveLength(3);
      
      const docIds = docIndices.map(idx => index.getDocumentId(idx));
      expect(docIds).toContain('doc1');
      expect(docIds).toContain('doc2');
      expect(docIds).toContain('doc3');
    });

    it('should perform early termination on empty intersection', () => {
      // Looking for trigrams where no document has all of them
      const docIndices = index.findDocumentsContainingAllTrigrams(['abc', 'efg']);
      
      // doc1 has 'abc' but not 'efg', doc3 has 'efg' but not 'abc'
      expect(docIndices).toHaveLength(0);
    });

    it('should handle complex multi-trigram intersections', () => {
      // Find documents with all three trigrams: 'cde', 'def', 'efg'
      const docIndices = index.findDocumentsContainingAllTrigrams(['cde', 'def', 'efg']);
      
      // Only doc3 contains all three
      expect(docIndices).toHaveLength(1);
      
      const docIds = docIndices.map(idx => index.getDocumentId(idx));
      expect(docIds).toContain('doc3');
    });
  });

  describe('Bitmap Trigram Union', () => {
    beforeEach(() => {
      const positions1: DocumentPosition[] = [
        { doc_id: 'doc1', file_path: '/test/file1.ts', line: 1, col: 0, length: 8 },
      ];
      const positions2: DocumentPosition[] = [
        { doc_id: 'doc2', file_path: '/test/file2.ts', line: 2, col: 0, length: 8 },
      ];

      index.addDocument('doc1', ['abc', 'def'], positions1);
      index.addDocument('doc2', ['ghi', 'jkl'], positions2);
    });

    it('should find documents containing any of the specified trigrams', () => {
      const docIndices = index.findDocumentsContainingAnyTrigram(['abc', 'ghi']);
      
      // Both documents should be found (doc1 has 'abc', doc2 has 'ghi')
      expect(docIndices).toHaveLength(2);
      
      const docIds = docIndices.map(idx => index.getDocumentId(idx));
      expect(docIds).toContain('doc1');
      expect(docIds).toContain('doc2');
    });

    it('should handle overlapping trigrams in union', () => {
      index.addDocument('doc3', ['abc', 'mno'], [
        { doc_id: 'doc3', file_path: '/test/file3.ts', line: 3, col: 0, length: 8 },
      ]);

      const docIndices = index.findDocumentsContainingAnyTrigram(['abc', 'def']);
      
      // doc1 contains both 'abc' and 'def', doc3 contains 'abc'
      expect(docIndices).toHaveLength(2);
      
      const docIds = docIndices.map(idx => index.getDocumentId(idx));
      expect(docIds).toContain('doc1');
      expect(docIds).toContain('doc3');
    });

    it('should return empty array for non-existent trigrams in union', () => {
      const docIndices = index.findDocumentsContainingAnyTrigram(['xyz', 'uvw']);
      expect(docIndices).toHaveLength(0);
    });
  });

  describe('Document Position Retrieval', () => {
    beforeEach(() => {
      const positions1: DocumentPosition[] = [
        { doc_id: 'doc1', file_path: '/test/file1.ts', line: 1, col: 0, length: 5 },
        { doc_id: 'doc1', file_path: '/test/file1.ts', line: 2, col: 5, length: 3 },
      ];
      const positions2: DocumentPosition[] = [
        { doc_id: 'doc2', file_path: '/test/file2.ts', line: 1, col: 10, length: 7 },
      ];

      index.addDocument('doc1', ['abc'], positions1);
      index.addDocument('doc2', ['def'], positions2);
    });

    it('should retrieve document positions by indices', () => {
      const docIndex1 = index.getDocumentIndex('doc1');
      const docIndex2 = index.getDocumentIndex('doc2');

      const positions = index.getDocumentPositions([docIndex1!, docIndex2!]);
      
      expect(positions).toHaveLength(3); // 2 positions from doc1 + 1 from doc2
      expect(positions.filter(p => p.doc_id === 'doc1')).toHaveLength(2);
      expect(positions.filter(p => p.doc_id === 'doc2')).toHaveLength(1);
    });

    it('should handle empty document indices array', () => {
      const positions = index.getDocumentPositions([]);
      expect(positions).toHaveLength(0);
    });

    it('should handle non-existent document indices', () => {
      const positions = index.getDocumentPositions([999]);
      expect(positions).toHaveLength(0);
    });
  });

  describe('Document Removal', () => {
    beforeEach(() => {
      const positions: DocumentPosition[] = [
        { doc_id: 'doc1', file_path: '/test/file1.ts', line: 1, col: 0, length: 5 },
      ];

      index.addDocument('doc1', ['abc', 'bcd'], positions);
      index.addDocument('doc2', ['bcd', 'cde'], positions);
    });

    it('should remove documents and clean up bitmaps', () => {
      const removed = index.removeDocument('doc1');
      expect(removed).toBe(true);

      // doc1 should no longer be found
      expect(index.getDocumentIndex('doc1')).toBeUndefined();
      
      // Trigram 'abc' should no longer have any documents
      const docIndices = index.findDocumentsContainingAllTrigrams(['abc']);
      expect(docIndices).toHaveLength(0);
      
      // Trigram 'bcd' should still have doc2
      const docIndices2 = index.findDocumentsContainingAllTrigrams(['bcd']);
      expect(docIndices2).toHaveLength(1);
    });

    it('should return false when removing non-existent documents', () => {
      const removed = index.removeDocument('non-existent');
      expect(removed).toBe(false);
    });
  });

  describe('Index Statistics and Management', () => {
    it('should provide accurate statistics', () => {
      const positions: DocumentPosition[] = [
        { doc_id: 'doc1', file_path: '/test/file1.ts', line: 1, col: 0, length: 5 },
      ];

      index.addDocument('doc1', ['abc', 'bcd'], positions);
      index.addDocument('doc2', ['bcd', 'cde'], positions);

      const stats = index.getStats();
      expect(stats.trigram_count).toBe(3); // 'abc', 'bcd', 'cde'
      expect(stats.document_count).toBe(2);
      expect(stats.total_bitmap_entries).toBe(4); // 'abc'->1, 'bcd'->2, 'cde'->1
      expect(stats.next_doc_index).toBe(2);
    });

    it('should clear all data', () => {
      const positions: DocumentPosition[] = [
        { doc_id: 'doc1', file_path: '/test/file1.ts', line: 1, col: 0, length: 5 },
      ];

      index.addDocument('doc1', ['abc'], positions);
      
      let stats = index.getStats();
      expect(stats.document_count).toBe(1);
      
      index.clear();
      
      stats = index.getStats();
      expect(stats.document_count).toBe(0);
      expect(stats.trigram_count).toBe(0);
      expect(stats.next_doc_index).toBe(0);
    });

    it('should compact the index efficiently', () => {
      const positions: DocumentPosition[] = [
        { doc_id: 'doc1', file_path: '/test/file1.ts', line: 1, col: 0, length: 5 },
      ];

      // Add documents
      index.addDocument('doc1', ['abc'], positions);
      index.addDocument('doc2', ['def'], positions);
      index.addDocument('doc3', ['ghi'], positions);

      // Remove middle document
      index.removeDocument('doc2');

      // Compact to resequence indices
      index.compact();

      // Should have 2 documents with sequential indices 0 and 1
      const stats = index.getStats();
      expect(stats.document_count).toBe(2);
      expect(stats.next_doc_index).toBe(2);

      // Verify documents are still accessible
      const doc1Index = index.getDocumentIndex('doc1');
      const doc3Index = index.getDocumentIndex('doc3');
      expect(doc1Index).toBeDefined();
      expect(doc3Index).toBeDefined();
      expect(doc1Index).not.toBe(doc3Index);
    });
  });

  describe('Performance Characteristics', () => {
    it('should handle large document sets efficiently', () => {
      const numDocs = 1000;
      const start = Date.now();

      // Add many documents with overlapping trigrams
      for (let i = 0; i < numDocs; i++) {
        const positions: DocumentPosition[] = [
          { doc_id: `doc${i}`, file_path: `/test/file${i}.ts`, line: 1, col: 0, length: 5 },
        ];
        
        const trigrams = [`t${i % 100}`, `t${(i + 1) % 100}`, `t${(i + 2) % 100}`];
        index.addDocument(`doc${i}`, trigrams, positions);
      }

      const indexTime = Date.now() - start;
      
      // Test search performance
      const searchStart = Date.now();
      const results = index.findDocumentsContainingAllTrigrams(['t0', 't1']);
      const searchTime = Date.now() - searchStart;

      // Performance assertions
      expect(indexTime).toBeLessThan(1000); // Indexing should be fast
      expect(searchTime).toBeLessThan(10); // Search should be very fast
      expect(results.length).toBeGreaterThan(0); // Should find matches
      
      const stats = index.getStats();
      expect(stats.document_count).toBe(numDocs);
      expect(stats.memory_efficiency).toBeGreaterThan(0.3); // Should be memory efficient
    });

    it('should demonstrate bitmap intersection efficiency', () => {
      // Create overlapping trigram patterns that would be expensive with Set operations
      const numDocs = 500;
      const commonTrigrams = ['common1', 'common2', 'common3'];

      for (let i = 0; i < numDocs; i++) {
        const positions: DocumentPosition[] = [
          { doc_id: `doc${i}`, file_path: `/test/file${i}.ts`, line: 1, col: 0, length: 5 },
        ];
        
        // Each document has common trigrams plus some unique ones
        const trigrams = [...commonTrigrams, `unique${i}`];
        index.addDocument(`doc${i}`, trigrams, positions);
      }

      // Test intersection performance on common trigrams
      const start = Date.now();
      const results = index.findDocumentsContainingAllTrigrams(commonTrigrams);
      const duration = Date.now() - start;

      expect(results).toHaveLength(numDocs); // All documents match
      expect(duration).toBeLessThan(5); // Should be very fast with bitmaps
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty trigram lists', () => {
      const docIndices1 = index.findDocumentsContainingAllTrigrams([]);
      const docIndices2 = index.findDocumentsContainingAnyTrigram([]);
      
      expect(docIndices1).toHaveLength(0);
      expect(docIndices2).toHaveLength(0);
    });

    it('should handle undefined trigrams in arrays', () => {
      const positions: DocumentPosition[] = [
        { doc_id: 'doc1', file_path: '/test/file1.ts', line: 1, col: 0, length: 5 },
      ];

      index.addDocument('doc1', ['abc', 'def'], positions);

      // Should handle arrays with undefined values gracefully
      const trigrams = ['abc', undefined, 'def'] as any[];
      const docIndices = index.findDocumentsContainingAllTrigrams(trigrams);
      
      expect(docIndices).toHaveLength(1);
    });

    it('should handle duplicate document additions', () => {
      const positions: DocumentPosition[] = [
        { doc_id: 'doc1', file_path: '/test/file1.ts', line: 1, col: 0, length: 5 },
      ];

      index.addDocument('doc1', ['abc'], positions);
      index.addDocument('doc1', ['def'], positions); // Same doc ID

      // Should reuse the same document index
      const stats = index.getStats();
      expect(stats.document_count).toBe(1);
      expect(stats.next_doc_index).toBe(1);
      
      // Should be findable by either trigram
      const docIndices1 = index.findDocumentsContainingAllTrigrams(['abc']);
      const docIndices2 = index.findDocumentsContainingAllTrigrams(['def']);
      
      expect(docIndices1).toHaveLength(1);
      expect(docIndices2).toHaveLength(1);
    });
  });
});