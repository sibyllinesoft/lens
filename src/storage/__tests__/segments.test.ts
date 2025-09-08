/**
 * Unit Tests for Segment Storage System
 * Tests memory-mapped segment storage with filesystem mocking
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { SegmentStorage } from '../segments.js';
import type { SegmentType } from '../../types/core.js';

// Mock filesystem operations
vi.mock('fs');
vi.mock('path');

// Mock telemetry
vi.mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn(() => ({
      setAttributes: vi.fn(),
      end: vi.fn(),
    })),
  },
}));

// Mock production config
vi.mock('../../types/config.js', () => ({
  PRODUCTION_CONFIG: {
    segments: {
      defaultSize: 16 * 1024 * 1024,
    },
  },
}));

describe('SegmentStorage', () => {
  let storage: SegmentStorage;
  let mockFs: any;

  beforeEach(() => {
    mockFs = {
      existsSync: vi.fn(),
      mkdirSync: vi.fn(),
      openSync: vi.fn(),
      ftruncateSync: vi.fn(),
      readSync: vi.fn(),
      writeSync: vi.fn(),
      closeSync: vi.fn(),
      unlinkSync: vi.fn(),
    };

    // Set up default mock behaviors
    mockFs.existsSync.mockReturnValue(false);
    mockFs.openSync.mockReturnValue(3); // Mock file descriptor
    mockFs.ftruncateSync.mockReturnValue(undefined);
    mockFs.readSync.mockReturnValue(64);
    mockFs.writeSync.mockReturnValue(64);

    // Apply mocks
    Object.keys(mockFs).forEach(key => {
      (fs as any)[key] = mockFs[key];
    });

    storage = new SegmentStorage('./test-segments');
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.restoreAllMocks();
  });

  describe('Constructor', () => {
    it('should initialize with default base path', () => {
      const defaultStorage = new SegmentStorage();
      expect(defaultStorage).toBeInstanceOf(SegmentStorage);
    });

    it('should initialize with custom base path', () => {
      const customStorage = new SegmentStorage('./custom-path');
      expect(customStorage).toBeInstanceOf(SegmentStorage);
    });

    it('should not create directory immediately', () => {
      new SegmentStorage('./lazy-path');
      expect(mockFs.mkdirSync).not.toHaveBeenCalled();
    });
  });

  describe('Directory Management', () => {
    it('should create directory when needed for first segment', async () => {
      mockFs.existsSync.mockReturnValue(false);

      await storage.createSegment('test-seg', 'lexical');

      expect(mockFs.existsSync).toHaveBeenCalledWith('./test-segments');
      expect(mockFs.mkdirSync).toHaveBeenCalledWith('./test-segments', { recursive: true });
    });

    it('should not create directory if it already exists', async () => {
      mockFs.existsSync.mockReturnValue(true);

      await storage.createSegment('test-seg', 'lexical');

      expect(mockFs.existsSync).toHaveBeenCalledWith('./test-segments');
      expect(mockFs.mkdirSync).not.toHaveBeenCalled();
    });
  });

  describe('Segment Creation', () => {
    it('should create a new segment with default size', async () => {
      const segmentId = 'test-segment';
      const segmentType: SegmentType = 'lexical';

      const segment = await storage.createSegment(segmentId, segmentType);

      expect(segment).toBeDefined();
      expect(segment.file_path).toBe('./test-segments/test-segment.lexical.seg');
      expect(segment.size).toBe(16 * 1024 * 1024); // Default 16MB
      expect(segment.readonly).toBe(false);
      expect(segment.buffer).toBeInstanceOf(Buffer);
      expect(segment.fd).toBe(3); // Mock file descriptor
    });

    it('should create a segment with custom size', async () => {
      const customSize = 8 * 1024 * 1024; // 8MB
      const segment = await storage.createSegment('custom-seg', 'semantic', customSize);

      expect(segment.size).toBe(customSize);
      expect(mockFs.ftruncateSync).toHaveBeenCalledWith(3, customSize);
    });

    it('should create segments with different types', async () => {
      const types: SegmentType[] = ['lexical', 'semantic', 'symbols', 'ngram'];

      for (const type of types) {
        mockFs.openSync.mockReturnValue(3 + types.indexOf(type)); // Different FD for each

        const segment = await storage.createSegment(`seg-${type}`, type);
        expect(segment.file_path).toContain(`.${type}.seg`);
      }
    });

    it('should write correct header to segment file', async () => {
      await storage.createSegment('header-test', 'lexical');

      expect(mockFs.writeSync).toHaveBeenCalled();
      
      // Verify writeHeader was called with correct structure
      const writeCall = mockFs.writeSync.mock.calls[0];
      expect(writeCall[0]).toBe(3); // file descriptor
      expect(writeCall[1]).toBeInstanceOf(Buffer); // header buffer
      expect(writeCall[2]).toBe(0); // offset
      expect(writeCall[3]).toBe(32); // header size
      expect(writeCall[4]).toBe(0); // file position
    });

    it('should throw error if segment already exists', async () => {
      mockFs.existsSync
        .mockReturnValueOnce(true) // Directory exists
        .mockReturnValueOnce(true); // File exists

      await expect(storage.createSegment('duplicate', 'lexical'))
        .rejects.toThrow('Segment duplicate already exists');
    });
  });

  describe('Segment Loading', () => {
    beforeEach(() => {
      // Mock successful file operations for loading
      mockFs.existsSync.mockImplementation((filePath: string) => {
        return !filePath.endsWith('nonexistent.seg');
      });
      
      // Mock header reading
      mockFs.readSync.mockImplementation((fd: number, buffer: Buffer, offset: number, length: number, position: number) => {
        if (position === 0 && length === 32) {
          // Mock reading header
          const header = Buffer.alloc(32);
          header.writeUInt32LE(0x4C454E53, 0); // Magic 'LENS'
          header.writeUInt32LE(1, 4); // Version
          header.writeUInt32LE(0, 8); // Type (lexical = 0)
          header.writeUInt32LE(16 * 1024 * 1024, 12); // Size
          header.writeUInt32LE(0, 16); // Checksum
          header.copy(buffer, offset);
        }
        return length;
      });
    });

    it('should load existing segment', async () => {
      const segment = await storage.loadSegment('existing-seg', 'lexical');

      expect(segment).toBeDefined();
      expect(segment.file_path).toBe('./test-segments/existing-seg.lexical.seg');
      expect(segment.readonly).toBe(true);
      expect(mockFs.openSync).toHaveBeenCalledWith('./test-segments/existing-seg.lexical.seg', 'r');
    });

    it('should throw error when loading nonexistent segment', async () => {
      await expect(storage.loadSegment('nonexistent', 'lexical'))
        .rejects.toThrow('Segment file ./test-segments/nonexistent.lexical.seg does not exist');
    });

    it('should validate magic number in header', async () => {
      // Mock invalid magic number
      mockFs.readSync.mockImplementation((fd: number, buffer: Buffer) => {
        const header = Buffer.alloc(32);
        header.writeUInt32LE(0x12345678, 0); // Invalid magic
        header.copy(buffer);
        return 32;
      });

      await expect(storage.loadSegment('invalid-magic', 'lexical'))
        .rejects.toThrow('Invalid segment header: magic number mismatch');
    });
  });

  describe('Segment Operations', () => {
    let testSegment: any;

    beforeEach(async () => {
      testSegment = await storage.createSegment('ops-test', 'lexical');
    });

    it('should write data to segment', async () => {
      const data = Buffer.from('test data');
      const bytesWritten = await storage.writeToSegment('ops-test', data, 64);

      expect(bytesWritten).toBe(data.length);
      expect(mockFs.writeSync).toHaveBeenCalledWith(
        testSegment.fd,
        data,
        0,
        data.length,
        64
      );
    });

    it('should read data from segment', async () => {
      const buffer = Buffer.alloc(10);
      
      // Mock reading data
      mockFs.readSync.mockReturnValue(10);
      
      const bytesRead = await storage.readFromSegment('ops-test', buffer, 64, 10);

      expect(bytesRead).toBe(10);
      expect(mockFs.readSync).toHaveBeenCalledWith(
        testSegment.fd,
        buffer,
        0,
        10,
        64
      );
    });

    it('should handle read/write errors gracefully', async () => {
      // Mock filesystem error
      mockFs.writeSync.mockImplementation(() => {
        throw new Error('Disk full');
      });

      const data = Buffer.from('test data');
      
      await expect(storage.writeToSegment('ops-test', data, 64))
        .rejects.toThrow('Failed to write to segment ops-test: Disk full');
    });
  });

  describe('Segment Cleanup', () => {
    beforeEach(async () => {
      await storage.createSegment('cleanup-test', 'lexical');
    });

    it('should close segment', async () => {
      await storage.closeSegment('cleanup-test');

      expect(mockFs.closeSync).toHaveBeenCalledWith(3);
    });

    it('should delete segment file', async () => {
      mockFs.existsSync.mockReturnValue(true);
      
      await storage.deleteSegment('cleanup-test');

      expect(mockFs.unlinkSync).toHaveBeenCalledWith('./test-segments/cleanup-test.lexical.seg');
    });

    it('should handle deletion of nonexistent segment', async () => {
      mockFs.existsSync.mockReturnValue(false);
      
      await expect(storage.deleteSegment('nonexistent'))
        .rejects.toThrow('Cannot delete segment nonexistent: not found');
    });

    it('should close all segments', async () => {
      await storage.createSegment('seg1', 'semantic');
      await storage.createSegment('seg2', 'symbols');

      await storage.closeAllSegments();

      expect(mockFs.closeSync).toHaveBeenCalledTimes(3); // cleanup-test + seg1 + seg2
    });
  });

  describe('Segment Information', () => {
    beforeEach(async () => {
      await storage.createSegment('info-test', 'lexical');
    });

    it('should get segment info', () => {
      const info = storage.getSegmentInfo('info-test');

      expect(info).toBeDefined();
      expect(info.segmentId).toBe('info-test');
      expect(info.type).toBe('lexical');
      expect(info.size).toBe(16 * 1024 * 1024);
      expect(info.readonly).toBe(false);
    });

    it('should return undefined for nonexistent segment', () => {
      const info = storage.getSegmentInfo('nonexistent');
      expect(info).toBeUndefined();
    });

    it('should list all segments', () => {
      const segments = storage.listSegments();
      expect(segments).toEqual(['info-test']);
    });

    it('should get storage statistics', async () => {
      await storage.createSegment('stats-test', 'semantic', 8 * 1024 * 1024);
      
      const stats = storage.getStorageStats();

      expect(stats.totalSegments).toBe(2);
      expect(stats.totalSize).toBe(24 * 1024 * 1024); // 16MB + 8MB
      expect(stats.types).toEqual({
        lexical: 1,
        semantic: 1,
        symbols: 0,
        ngram: 0,
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle filesystem errors during creation', async () => {
      mockFs.openSync.mockImplementation(() => {
        throw new Error('Permission denied');
      });

      await expect(storage.createSegment('error-test', 'lexical'))
        .rejects.toThrow('Failed to create segment error-test: Permission denied');
    });

    it('should handle buffer allocation errors', async () => {
      // Mock Buffer.alloc to throw
      const originalAlloc = Buffer.alloc;
      Buffer.alloc = vi.fn().mockImplementation(() => {
        throw new Error('Out of memory');
      });

      await expect(storage.createSegment('memory-error', 'lexical'))
        .rejects.toThrow();

      Buffer.alloc = originalAlloc;
    });

    it('should validate segment parameters', async () => {
      await expect(storage.createSegment('', 'lexical'))
        .rejects.toThrow('Segment ID cannot be empty');

      await expect(storage.createSegment('test', 'lexical', -1))
        .rejects.toThrow('Initial size must be positive');

      await expect(storage.createSegment('test', 'lexical', 0))
        .rejects.toThrow('Initial size must be positive');
    });
  });

  describe('Memory Management', () => {
    it('should track memory usage', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      await storage.createSegment('memory-test', 'lexical', 1024 * 1024); // 1MB
      
      const info = storage.getSegmentInfo('memory-test');
      expect(info?.size).toBe(1024 * 1024);
    });

    it('should release memory on segment closure', async () => {
      await storage.createSegment('release-test', 'lexical');
      await storage.closeSegment('release-test');

      const info = storage.getSegmentInfo('release-test');
      expect(info).toBeUndefined();
    });
  });
});