/**
 * Unit tests for Segment Storage System
 * Tests memory-mapped segment operations, headers, and lifecycle
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SegmentStorage } from '../../src/storage/segments.js';
import * as fs from 'fs';
import * as path from 'path';

describe('SegmentStorage', () => {
  let storage: SegmentStorage;
  const testDir = './test-segments-unit';

  beforeEach(() => {
    // Clean up test directory
    if (fs.existsSync(testDir)) {
      fs.rmSync(testDir, { recursive: true, force: true });
    }
    storage = new SegmentStorage(testDir);
  });

  afterEach(async () => {
    await storage.shutdown();
    if (fs.existsSync(testDir)) {
      fs.rmSync(testDir, { recursive: true, force: true });
    }
  });

  describe('Segment Creation', () => {
    it('should create a new segment', async () => {
      const segmentId = 'test-segment-1';
      const segment = await storage.createSegment(segmentId, 'lexical', 1024);
      
      expect(segment.file_path).toContain(`${segmentId}.lexical.seg`);
      expect(segment.size).toBe(1024);
      expect(segment.readonly).toBe(false);
      expect(Buffer.isBuffer(segment.buffer)).toBe(true);
    });

    it('should create directory if it does not exist', async () => {
      const newDir = './new-test-segments';
      const newStorage = new SegmentStorage(newDir);
      
      try {
        await newStorage.createSegment('test', 'symbols', 512);
        expect(fs.existsSync(newDir)).toBe(true);
      } finally {
        await newStorage.shutdown();
        if (fs.existsSync(newDir)) {
          fs.rmSync(newDir, { recursive: true, force: true });
        }
      }
    });

    it('should fail when creating duplicate segment', async () => {
      const segmentId = 'duplicate-test';
      
      await storage.createSegment(segmentId, 'ast', 512);
      
      await expect(storage.createSegment(segmentId, 'ast', 512))
        .rejects.toThrow('already exists');
    });

    it('should handle different segment types', async () => {
      const types: Array<'lexical' | 'symbols' | 'ast' | 'semantic'> = ['lexical', 'symbols', 'ast', 'semantic'];
      
      for (let i = 0; i < types.length; i++) {
        const segment = await storage.createSegment(`test-${i}`, types[i], 1024);
        expect(segment.file_path).toContain(`.${types[i]}.seg`);
      }
    });
  });

  describe('Segment Opening', () => {
    beforeEach(async () => {
      // Create a test segment
      await storage.createSegment('existing-segment', 'lexical', 2048);
    });

    it('should open existing segment', async () => {
      const segment = await storage.openSegment('existing-segment');
      
      expect(segment.size).toBe(2048);
      expect(segment.readonly).toBe(false);
    });

    it('should open segment in readonly mode', async () => {
      const segment = await storage.openSegment('existing-segment', true);
      
      expect(segment.readonly).toBe(true);
    });

    it('should return cached segment on second open', async () => {
      const segment1 = await storage.openSegment('existing-segment');
      const segment2 = await storage.openSegment('existing-segment');
      
      expect(segment1).toBe(segment2);
    });

    it('should fail when opening non-existent segment', async () => {
      await expect(storage.openSegment('non-existent'))
        .rejects.toThrow('not found');
    });
  });

  describe('Segment Read/Write', () => {
    let segmentId: string;

    beforeEach(async () => {
      segmentId = 'read-write-test';
      await storage.createSegment(segmentId, 'lexical', 4096);
    });

    it('should write and read data', async () => {
      const testData = Buffer.from('Hello, Lens!', 'utf8');
      const offset = 1000;

      await storage.writeToSegment(segmentId, offset, testData);
      const readData = await storage.readFromSegment(segmentId, offset, testData.length);
      
      expect(readData.equals(testData)).toBe(true);
    });

    it('should write binary data correctly', async () => {
      const binaryData = Buffer.from([0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD]);
      const offset = 2000;

      await storage.writeToSegment(segmentId, offset, binaryData);
      const readData = await storage.readFromSegment(segmentId, offset, binaryData.length);
      
      expect(readData.equals(binaryData)).toBe(true);
    });

    it('should fail write on readonly segment', async () => {
      await storage.closeSegment(segmentId);
      await storage.openSegment(segmentId, true);
      
      const testData = Buffer.from('test');
      
      await expect(storage.writeToSegment(segmentId, 100, testData))
        .rejects.toThrow('read-only');
    });

    it('should fail on out-of-bounds write', async () => {
      const testData = Buffer.from('test data');
      const badOffset = 5000; // Beyond segment size
      
      await expect(storage.writeToSegment(segmentId, badOffset, testData))
        .rejects.toThrow('exceed segment size');
    });

    it('should fail on out-of-bounds read', async () => {
      const badOffset = 5000;
      
      await expect(storage.readFromSegment(segmentId, badOffset, 10))
        .rejects.toThrow('exceed segment size');
    });
  });

  describe('Segment Expansion', () => {
    let segmentId: string;

    beforeEach(async () => {
      segmentId = 'expansion-test';
      await storage.createSegment(segmentId, 'symbols', 1024);
    });

    it('should expand segment size', async () => {
      await storage.expandSegment(segmentId, 1024);
      
      const info = await storage.getSegmentInfo(segmentId);
      expect(info.size_bytes).toBe(2048);
    });

    it('should allow write after expansion', async () => {
      await storage.expandSegment(segmentId, 1024);
      
      const testData = Buffer.from('data after expansion');
      const offset = 1500; // In the expanded area
      
      await storage.writeToSegment(segmentId, offset, testData);
      const readData = await storage.readFromSegment(segmentId, offset, testData.length);
      
      expect(readData.equals(testData)).toBe(true);
    });

    it('should fail expansion on readonly segment', async () => {
      await storage.closeSegment(segmentId);
      await storage.openSegment(segmentId, true);
      
      await expect(storage.expandSegment(segmentId, 512))
        .rejects.toThrow('read-only');
    });

    it('should enforce size limits', async () => {
      // Try to expand beyond the limit
      const hugeSize = 1024 * 1024 * 1024; // 1GB
      
      await expect(storage.expandSegment(segmentId, hugeSize))
        .rejects.toThrow('exceed size limit');
    });
  });

  describe('Segment Information', () => {
    let segmentId: string;

    beforeEach(async () => {
      segmentId = 'info-test';
      await storage.createSegment(segmentId, 'ast', 2048);
    });

    it('should get segment information', async () => {
      const info = await storage.getSegmentInfo(segmentId);
      
      expect(info.id).toBe(segmentId);
      expect(info.type).toBe('ast');
      expect(info.size_bytes).toBe(2048);
      expect(info.memory_mapped).toBe(true);
      expect(info.last_accessed).toBeInstanceOf(Date);
    });

    it('should list all segments', async () => {
      await storage.createSegment('segment1', 'lexical', 1024);
      await storage.createSegment('segment2', 'symbols', 1024);
      
      const segments = storage.listSegments();
      
      expect(segments).toContain('info-test');
      expect(segments).toContain('segment1');
      expect(segments).toContain('segment2');
      expect(segments.length).toBe(3);
    });

    it('should handle non-existent segment info', async () => {
      await expect(storage.getSegmentInfo('non-existent'))
        .rejects.toThrow('not found');
    });
  });

  describe('Segment Closure', () => {
    let segmentId: string;

    beforeEach(async () => {
      segmentId = 'closure-test';
      await storage.createSegment(segmentId, 'semantic', 1024);
    });

    it('should close segment', async () => {
      await storage.closeSegment(segmentId);
      
      const info = await storage.getSegmentInfo(segmentId);
      expect(info.memory_mapped).toBe(false);
    });

    it('should handle closing non-open segment', async () => {
      await storage.closeSegment('non-existent');
      
      // Should not throw error
      expect(true).toBe(true);
    });

    it('should allow reopening after close', async () => {
      // Write some data first
      const testData = Buffer.from('test before close');
      await storage.writeToSegment(segmentId, 100, testData);
      
      // Close and reopen
      await storage.closeSegment(segmentId);
      await storage.openSegment(segmentId);
      
      // Should still be able to read data
      const readData = await storage.readFromSegment(segmentId, 100, testData.length);
      expect(readData.equals(testData)).toBe(true);
    });
  });

  describe('Segment Compaction', () => {
    it('should compact segment', async () => {
      const segmentId = 'compact-test';
      await storage.createSegment(segmentId, 'lexical', 4096);
      
      // Compaction should not fail (even if it's a placeholder implementation)
      try {
        await storage.compactSegment(segmentId);
        // If we reach here, the method completed successfully
        expect(true).toBe(true);
      } catch (error) {
        // If we reach here, the method threw an error
        throw new Error(`compactSegment should not throw, but threw: ${error}`);
      }
    });
  });

  describe('Error Handling', () => {
    it('should handle file system errors gracefully', async () => {
      const invalidStorage = new SegmentStorage('/invalid/path/that/cannot/be/created');
      
      await expect(invalidStorage.createSegment('test', 'lexical', 1024))
        .rejects.toThrow();
    });

    it('should validate segment headers', async () => {
      const segmentId = 'header-test';
      await storage.createSegment(segmentId, 'lexical', 1024);
      await storage.closeSegment(segmentId);
      
      // Corrupt the header by writing invalid magic number
      const filePath = path.join(testDir, `${segmentId}.lexical.seg`);
      const fd = fs.openSync(filePath, 'r+');
      const corruptHeader = Buffer.alloc(4);
      corruptHeader.writeUInt32LE(0xDEADBEEF, 0); // Wrong magic
      fs.writeSync(fd, corruptHeader, 0, 4, 0);
      fs.closeSync(fd);
      
      // Should fail to open corrupted segment
      await expect(storage.openSegment(segmentId))
        .rejects.toThrow('Invalid segment magic');
    });
  });

  describe('Shutdown', () => {
    it('should shutdown cleanly', async () => {
      await storage.createSegment('test1', 'lexical', 1024);
      await storage.createSegment('test2', 'symbols', 1024);
      
      try {
        await storage.shutdown();
        expect(true).toBe(true);
      } catch (error) {
        throw new Error(`shutdown should not throw, but threw: ${error}`);
      }
    });

    it('should handle multiple shutdowns', async () => {
      await storage.shutdown();
      try {
        await storage.shutdown();
        expect(true).toBe(true);
      } catch (error) {
        throw new Error(`multiple shutdown should not throw, but threw: ${error}`);
      }
    });
  });
});