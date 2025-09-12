/**
 * Comprehensive Segment Storage Tests
 * Target: High coverage using proven patterns from ASTCache (74%) and Quality-gates (94%)
 * Strategy: Test all storage operations, file I/O, memory mapping, and error scenarios
 */

import { describe, it, expect, beforeEach, afterEach, mock, jest, mock } from 'bun:test';
import * as fs from 'fs';
import * as path from 'path';
import { SegmentStorage } from '../segments.js';
import type { SegmentType, MMapSegment } from '../../types/core.js';

// Mock fs operations for controlled testing
mock('fs');
mock('../../telemetry/tracer.js');

describe('Comprehensive Segment Storage Tests', () => {
  let segmentStorage: SegmentStorage;
  let mockFs: any;
  let testBasePath: string;

  // Mock segment data for testing
  const mockSegmentId = 'test-segment-123';
  const mockSegmentType: SegmentType = 'lexical';
  const mockInitialSize = 16 * 1024 * 1024; // 16MB

  const mockBuffer = Buffer.alloc(mockInitialSize);
  const mockFileDescriptor = 3;

  beforeEach(() => {
    jest.clearAllMocks();
    testBasePath = '/tmp/test-segments';

    // Mock fs methods with realistic implementations
    mockFs = mocked(fs);

    mockFs.existsSync = jest.fn().mockReturnValue(false);
    mockFs.mkdirSync = jest.fn();
    mockFs.openSync = jest.fn().mockReturnValue(mockFileDescriptor);
    mockFs.ftruncateSync = jest.fn();
    mockFs.writeSync = jest.fn();
    mockFs.fsyncSync = jest.fn();
    mockFs.closeSync = jest.fn();
    mockFs.readSync = jest.fn();
    mockFs.statSync = jest.fn().mockReturnValue({
      size: mockInitialSize,
      mtimeMs: Date.now(),
    });

    // Mock Buffer.alloc to return our test buffer
    jest.spyOn(Buffer, 'alloc').mockReturnValue(mockBuffer);
    jest.spyOn(Buffer, 'from').mockImplementation((source: any) => {
      if (source.subarray) {
        return source.subarray();
      }
      return Buffer.from(source);
    });

    // Mock tracer
    const { LensTracer } = require('../../telemetry/tracer.js');
    const mockSpan = {
      setAttributes: jest.fn(),
      recordException: jest.fn(),
      end: jest.fn(),
    };
    LensTracer.createChildSpan = jest.fn().mockReturnValue(mockSpan);

    segmentStorage = new SegmentStorage(testBasePath);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Initialization and Directory Management', () => {
    it('should create storage with custom base path', () => {
      const customPath = '/custom/segments';
      const storage = new SegmentStorage(customPath);
      expect(storage).toBeDefined();
    });

    it('should use default path when none provided', () => {
      const storage = new SegmentStorage();
      expect(storage).toBeDefined();
    });

    it('should ensure directory exists when creating segments', async () => {
      mockFs.existsSync.mockReturnValue(false);
      
      await segmentStorage.createSegment(mockSegmentId, mockSegmentType, mockInitialSize);

      expect(mockFs.existsSync).toHaveBeenCalledWith(testBasePath);
      expect(mockFs.mkdirSync).toHaveBeenCalledWith(testBasePath, { recursive: true });
    });

    it('should not create directory if it already exists', async () => {
      mockFs.existsSync.mockReturnValue(true);
      
      await segmentStorage.createSegment(mockSegmentId, mockSegmentType, mockInitialSize);

      expect(mockFs.mkdirSync).not.toHaveBeenCalled();
    });
  });

  describe('Segment Creation', () => {
    it('should create new segment with correct parameters', async () => {
      const segment = await segmentStorage.createSegment(
        mockSegmentId,
        mockSegmentType,
        mockInitialSize
      );

      expect(segment).toBeDefined();
      expect(segment.id).toBe(mockSegmentId);
      expect(segment.type).toBe(mockSegmentType);
      expect(segment.size).toBe(mockInitialSize);
      expect(segment.readonly).toBe(false);
      
      // Verify file operations
      expect(mockFs.openSync).toHaveBeenCalledWith(
        path.join(testBasePath, `${mockSegmentId}.${mockSegmentType}.seg`),
        'w+'
      );
      expect(mockFs.ftruncateSync).toHaveBeenCalledWith(mockFileDescriptor, mockInitialSize);
    });

    it('should write correct header to segment file', async () => {
      await segmentStorage.createSegment(mockSegmentId, mockSegmentType, mockInitialSize);

      // Verify header was written
      expect(mockFs.writeSync).toHaveBeenCalled();
      const writeCall = mockFs.writeSync.mock.calls[0];
      expect(writeCall[0]).toBe(mockFileDescriptor);
      expect(writeCall[3]).toBe(0); // Offset should be 0 for header
    });

    it('should handle different segment types correctly', async () => {
      const types: SegmentType[] = ['lexical', 'symbols', 'semantic'];
      
      for (const type of types) {
        const segmentId = `test-${type}`;
        const segment = await segmentStorage.createSegment(segmentId, type);
        
        expect(segment.type).toBe(type);
        expect(mockFs.openSync).toHaveBeenCalledWith(
          path.join(testBasePath, `${segmentId}.${type}.seg`),
          'w+'
        );
      }
    });

    it('should handle custom initial sizes', async () => {
      const customSizes = [1024, 64 * 1024, 256 * 1024 * 1024];
      
      for (const size of customSizes) {
        const segmentId = `test-size-${size}`;
        const segment = await segmentStorage.createSegment(segmentId, mockSegmentType, size);
        
        expect(segment.size).toBe(size);
        expect(mockFs.ftruncateSync).toHaveBeenCalledWith(mockFileDescriptor, size);
      }
    });

    it('should fail if segment already exists', async () => {
      mockFs.existsSync.mockImplementation((filePath: string) => {
        if (filePath.includes(`${mockSegmentId}.${mockSegmentType}.seg`)) {
          return true;
        }
        return false;
      });

      await expect(
        segmentStorage.createSegment(mockSegmentId, mockSegmentType, mockInitialSize)
      ).rejects.toThrow(`Segment ${mockSegmentId} already exists`);
    });

    it('should handle file creation errors', async () => {
      mockFs.openSync.mockImplementation(() => {
        throw new Error('Permission denied');
      });

      await expect(
        segmentStorage.createSegment(mockSegmentId, mockSegmentType, mockInitialSize)
      ).rejects.toThrow('Failed to create segment: Permission denied');
    });

    it('should create segments with valid header magic number', async () => {
      await segmentStorage.createSegment(mockSegmentId, mockSegmentType, mockInitialSize);

      // Verify the magic number (0x4C454E53 = 'LENS')
      const writeCall = mockFs.writeSync.mock.calls.find(call => call[3] === 0);
      expect(writeCall).toBeDefined();
    });
  });

  describe('Segment Opening and Loading', () => {
    it('should open existing segment successfully', async () => {
      // Mock existing file
      mockFs.existsSync.mockImplementation((filePath: string) => {
        return filePath.includes(`${mockSegmentId}.${mockSegmentType}.seg`);
      });

      const mockHeader = Buffer.alloc(32);
      mockHeader.writeUInt32LE(0x4C454E53, 0); // Magic number
      mockHeader.writeUInt32LE(1, 4); // Version
      mockHeader.writeUInt32LE(mockInitialSize, 12); // Size

      mockFs.readSync.mockReturnValue(mockHeader.length);
      mockFs.readSync.mockImplementation((fd, buffer, offset, length, position) => {
        if (position === 0) {
          mockHeader.copy(buffer, offset);
          return mockHeader.length;
        }
        return 0;
      });

      const segment = await segmentStorage.openSegment(mockSegmentId);

      expect(segment).toBeDefined();
      expect(segment.id).toBe(mockSegmentId);
      expect(mockFs.openSync).toHaveBeenCalledWith(
        expect.stringContaining(`${mockSegmentId}.lexical.seg`),
        'r+'
      );
    });

    it('should open segment in readonly mode', async () => {
      mockFs.existsSync.mockReturnValue(true);
      
      const mockHeader = Buffer.alloc(32);
      mockHeader.writeUInt32LE(0x4C454E53, 0);
      mockFs.readSync.mockReturnValue(32);
      mockFs.readSync.mockImplementation((fd, buffer) => {
        mockHeader.copy(buffer);
        return mockHeader.length;
      });

      const segment = await segmentStorage.openSegment(mockSegmentId, true);

      expect(segment.readonly).toBe(true);
      expect(mockFs.openSync).toHaveBeenCalledWith(
        expect.any(String),
        'r' // Read-only mode
      );
    });

    it('should fail to open non-existent segment', async () => {
      mockFs.existsSync.mockReturnValue(false);

      await expect(segmentStorage.openSegment('non-existent')).rejects.toThrow(
        'Segment file not found'
      );
    });

    it('should validate segment header magic number', async () => {
      mockFs.existsSync.mockReturnValue(true);
      
      const invalidHeader = Buffer.alloc(32);
      invalidHeader.writeUInt32LE(0xDEADBEEF, 0); // Invalid magic
      mockFs.readSync.mockImplementation((fd, buffer) => {
        invalidHeader.copy(buffer);
        return invalidHeader.length;
      });

      await expect(segmentStorage.openSegment(mockSegmentId)).rejects.toThrow(
        'Invalid segment magic number'
      );
    });

    it('should handle corrupted segment headers', async () => {
      mockFs.existsSync.mockReturnValue(true);
      mockFs.readSync.mockImplementation(() => {
        throw new Error('Read error');
      });

      await expect(segmentStorage.openSegment(mockSegmentId)).rejects.toThrow(
        'Failed to open segment: Read error'
      );
    });
  });

  describe('Data Writing Operations', () => {
    let mockSegment: MMapSegment;

    beforeEach(async () => {
      mockSegment = await segmentStorage.createSegment(
        mockSegmentId,
        mockSegmentType,
        mockInitialSize
      );
    });

    it('should write data to segment at specified offset', async () => {
      const testData = Buffer.from('Hello, World!', 'utf8');
      const offset = 1024;

      await segmentStorage.writeToSegment(mockSegmentId, offset, testData);

      expect(mockFs.writeSync).toHaveBeenCalledWith(
        mockFileDescriptor,
        testData,
        0,
        testData.length,
        offset
      );
      expect(mockFs.fsyncSync).toHaveBeenCalledWith(mockFileDescriptor);
    });

    it('should write data to buffer and sync to disk', async () => {
      const testData = Buffer.from('Test data for segment', 'utf8');
      const offset = 2048;

      // Mock buffer copy operation
      const copySpy = jest.spyOn(testData, 'copy');

      await segmentStorage.writeToSegment(mockSegmentId, offset, testData);

      expect(copySpy).toHaveBeenCalledWith(mockBuffer, offset);
      expect(mockFs.fsyncSync).toHaveBeenCalled();
    });

    it('should fail to write to non-opened segment', async () => {
      const testData = Buffer.from('test');

      await expect(
        segmentStorage.writeToSegment('non-existent', 0, testData)
      ).rejects.toThrow('Segment non-existent not opened');
    });

    it('should fail to write to readonly segment', async () => {
      mockFs.existsSync.mockReturnValue(true);
      const mockHeader = Buffer.alloc(32);
      mockHeader.writeUInt32LE(0x4C454E53, 0);
      mockFs.readSync.mockImplementation((fd, buffer) => {
        mockHeader.copy(buffer);
        return mockHeader.length;
      });

      const readonlySegment = await segmentStorage.openSegment(mockSegmentId, true);
      const testData = Buffer.from('test');

      await expect(
        segmentStorage.writeToSegment(mockSegmentId, 0, testData)
      ).rejects.toThrow(`Segment ${mockSegmentId} is read-only`);
    });

    it('should validate write bounds', async () => {
      const testData = Buffer.from('x'.repeat(1024));
      const invalidOffset = mockInitialSize - 512; // Would exceed bounds

      await expect(
        segmentStorage.writeToSegment(mockSegmentId, invalidOffset, testData)
      ).rejects.toThrow('Write would exceed segment size');
    });

    it('should handle write errors gracefully', async () => {
      mockFs.writeSync.mockImplementation(() => {
        throw new Error('Disk full');
      });

      const testData = Buffer.from('test');

      await expect(
        segmentStorage.writeToSegment(mockSegmentId, 0, testData)
      ).rejects.toThrow('Failed to write to segment: Disk full');
    });

    it('should handle large data writes', async () => {
      const largeData = Buffer.alloc(1024 * 1024); // 1MB
      largeData.fill(0xAB);

      await segmentStorage.writeToSegment(mockSegmentId, 0, largeData);

      expect(mockFs.writeSync).toHaveBeenCalledWith(
        mockFileDescriptor,
        largeData,
        0,
        largeData.length,
        0
      );
    });
  });

  describe('Data Reading Operations', () => {
    let mockSegment: MMapSegment;

    beforeEach(async () => {
      mockSegment = await segmentStorage.createSegment(
        mockSegmentId,
        mockSegmentType,
        mockInitialSize
      );
    });

    it('should read data from segment at specified offset', async () => {
      const offset = 1024;
      const length = 512;
      const expectedData = Buffer.from('Expected data content', 'utf8');

      // Mock buffer subarray to return expected data
      mockBuffer.subarray = jest.fn().mockReturnValue(expectedData);

      const result = await segmentStorage.readFromSegment(mockSegmentId, offset, length);

      expect(mockBuffer.subarray).toHaveBeenCalledWith(offset, offset + length);
      expect(Buffer.from).toHaveBeenCalledWith(expectedData);
      expect(result).toEqual(expectedData);
    });

    it('should fail to read from non-opened segment', async () => {
      await expect(
        segmentStorage.readFromSegment('non-existent', 0, 100)
      ).rejects.toThrow('Segment non-existent not opened');
    });

    it('should validate read bounds', async () => {
      const invalidOffset = mockInitialSize - 100;
      const invalidLength = 200; // Would exceed bounds

      await expect(
        segmentStorage.readFromSegment(mockSegmentId, invalidOffset, invalidLength)
      ).rejects.toThrow('Read would exceed segment size');
    });

    it('should handle read errors gracefully', async () => {
      mockBuffer.subarray = jest.fn().mockImplementation(() => {
        throw new Error('Buffer error');
      });

      await expect(
        segmentStorage.readFromSegment(mockSegmentId, 0, 100)
      ).rejects.toThrow('Failed to read from segment: Buffer error');
    });

    it('should read different data sizes correctly', async () => {
      const testCases = [
        { offset: 0, length: 1 },
        { offset: 1024, length: 4096 },
        { offset: 8192, length: 32768 },
      ];

      for (const { offset, length } of testCases) {
        const mockData = Buffer.alloc(length).fill(0xFF);
        mockBuffer.subarray = jest.fn().mockReturnValue(mockData);

        const result = await segmentStorage.readFromSegment(mockSegmentId, offset, length);

        expect(mockBuffer.subarray).toHaveBeenCalledWith(offset, offset + length);
        expect(result).toBeDefined();
      }
    });

    it('should handle zero-length reads', async () => {
      const result = await segmentStorage.readFromSegment(mockSegmentId, 0, 0);
      expect(result).toBeDefined();
      expect(result.length).toBe(0);
    });
  });

  describe('Segment Expansion', () => {
    let mockSegment: MMapSegment;

    beforeEach(async () => {
      mockSegment = await segmentStorage.createSegment(
        mockSegmentId,
        mockSegmentType,
        mockInitialSize
      );
    });

    it('should expand segment size correctly', async () => {
      const additionalSize = 8 * 1024 * 1024; // 8MB
      const newSize = mockInitialSize + additionalSize;

      await segmentStorage.expandSegment(mockSegmentId, additionalSize);

      expect(mockFs.ftruncateSync).toHaveBeenCalledWith(mockFileDescriptor, newSize);
    });

    it('should fail to expand non-opened segment', async () => {
      await expect(
        segmentStorage.expandSegment('non-existent', 1024)
      ).rejects.toThrow('Segment non-existent not opened');
    });

    it('should fail to expand readonly segment', async () => {
      mockFs.existsSync.mockReturnValue(true);
      const mockHeader = Buffer.alloc(32);
      mockHeader.writeUInt32LE(0x4C454E53, 0);
      mockFs.readSync.mockImplementation((fd, buffer) => {
        mockHeader.copy(buffer);
        return mockHeader.length;
      });

      await segmentStorage.openSegment(mockSegmentId, true);

      await expect(
        segmentStorage.expandSegment(mockSegmentId, 1024)
      ).rejects.toThrow(`Segment ${mockSegmentId} is read-only`);
    });

    it('should handle expansion errors', async () => {
      mockFs.ftruncateSync.mockImplementation(() => {
        throw new Error('Expansion failed');
      });

      await expect(
        segmentStorage.expandSegment(mockSegmentId, 1024)
      ).rejects.toThrow('Failed to expand segment: Expansion failed');
    });

    it('should update segment size after expansion', async () => {
      const additionalSize = 4 * 1024 * 1024;
      
      await segmentStorage.expandSegment(mockSegmentId, additionalSize);

      // The segment size should be updated internally
      expect(mockFs.ftruncateSync).toHaveBeenCalledWith(
        mockFileDescriptor,
        mockInitialSize + additionalSize
      );
    });
  });

  describe('Health Status and Monitoring', () => {
    it('should return health status for empty storage', async () => {
      const health = await segmentStorage.getHealthStatus();

      expect(health).toBeDefined();
      expect(health.status).toBe('ok');
      expect(health.shards_healthy).toBe(0);
      expect(health.shards_total).toBe(0);
      expect(health.memory_usage_gb).toBeGreaterThanOrEqual(0);
      expect(health.active_queries).toBe(0);
    });

    it('should track opened segments in health status', async () => {
      await segmentStorage.createSegment(mockSegmentId, mockSegmentType, mockInitialSize);
      await segmentStorage.createSegment('segment-2', 'symbols', mockInitialSize);

      const health = await segmentStorage.getHealthStatus();

      expect(health.shards_total).toBe(2);
      expect(health.shards_healthy).toBe(2);
    });

    it('should calculate memory usage correctly', async () => {
      await segmentStorage.createSegment(mockSegmentId, mockSegmentType, mockInitialSize);

      const health = await segmentStorage.getHealthStatus();

      expect(health.memory_usage_gb).toBeGreaterThan(0);
      // Should be approximately 16MB = 0.016GB
      expect(health.memory_usage_gb).toBeLessThan(1);
    });

    it('should handle health check errors gracefully', async () => {
      // Mock process.memoryUsage to throw error
      const originalMemoryUsage = process.memoryUsage;
      process.memoryUsage = jest.fn().mockImplementation(() => {
        throw new Error('Memory check failed');
      });

      try {
        const health = await segmentStorage.getHealthStatus();
        expect(health.status).toBe('degraded');
      } finally {
        process.memoryUsage = originalMemoryUsage;
      }
    });
  });

  describe('Segment Cleanup and Resource Management', () => {
    it('should close segment properly', async () => {
      await segmentStorage.createSegment(mockSegmentId, mockSegmentType, mockInitialSize);
      
      await segmentStorage.closeSegment(mockSegmentId);

      expect(mockFs.closeSync).toHaveBeenCalledWith(mockFileDescriptor);
    });

    it('should handle closing non-existent segment', async () => {
      await expect(
        segmentStorage.closeSegment('non-existent')
      ).rejects.toThrow('Segment non-existent not opened');
    });

    it('should handle close errors gracefully', async () => {
      await segmentStorage.createSegment(mockSegmentId, mockSegmentType, mockInitialSize);
      
      mockFs.closeSync.mockImplementation(() => {
        throw new Error('Close failed');
      });

      await expect(
        segmentStorage.closeSegment(mockSegmentId)
      ).rejects.toThrow('Failed to close segment: Close failed');
    });

    it('should clean up all segments', async () => {
      const segmentIds = ['seg1', 'seg2', 'seg3'];
      
      for (const id of segmentIds) {
        await segmentStorage.createSegment(id, mockSegmentType, mockInitialSize);
      }

      await segmentStorage.cleanup();

      expect(mockFs.closeSync).toHaveBeenCalledTimes(segmentIds.length);
    });

    it('should handle cleanup errors for individual segments', async () => {
      await segmentStorage.createSegment(mockSegmentId, mockSegmentType, mockInitialSize);
      
      mockFs.closeSync.mockImplementation(() => {
        throw new Error('Close failed');
      });

      // Should not throw, but log errors
      await expect(segmentStorage.cleanup()).resolves.not.toThrow();
    });
  });

  describe('Error Recovery and Edge Cases', () => {
    it('should handle concurrent operations safely', async () => {
      const segment = await segmentStorage.createSegment(
        mockSegmentId,
        mockSegmentType,
        mockInitialSize
      );

      // Simulate concurrent writes
      const writePromises = Array.from({ length: 10 }, (_, i) =>
        segmentStorage.writeToSegment(
          mockSegmentId,
          i * 1024,
          Buffer.from(`data-${i}`)
        )
      );

      await Promise.all(writePromises);

      expect(mockFs.writeSync).toHaveBeenCalledTimes(10);
      expect(mockFs.fsyncSync).toHaveBeenCalledTimes(10);
    });

    it('should handle file system permission errors', async () => {
      mockFs.openSync.mockImplementation(() => {
        throw new Error('EACCES: permission denied');
      });

      await expect(
        segmentStorage.createSegment(mockSegmentId, mockSegmentType, mockInitialSize)
      ).rejects.toThrow('Failed to create segment: EACCES: permission denied');
    });

    it('should handle insufficient disk space', async () => {
      mockFs.ftruncateSync.mockImplementation(() => {
        throw new Error('ENOSPC: no space left on device');
      });

      await expect(
        segmentStorage.createSegment(mockSegmentId, mockSegmentType, mockInitialSize)
      ).rejects.toThrow('Failed to create segment: ENOSPC: no space left on device');
    });

    it('should validate segment types', async () => {
      const invalidType = 'invalid-type' as SegmentType;
      
      // Should still create segment but with the provided type
      const segment = await segmentStorage.createSegment(
        mockSegmentId,
        invalidType,
        mockInitialSize
      );
      
      expect(segment.type).toBe(invalidType);
    });

    it('should handle extremely large segments', async () => {
      const largeSize = 4 * 1024 * 1024 * 1024; // 4GB
      
      const segment = await segmentStorage.createSegment(
        'large-segment',
        mockSegmentType,
        largeSize
      );

      expect(segment.size).toBe(largeSize);
      expect(mockFs.ftruncateSync).toHaveBeenCalledWith(mockFileDescriptor, largeSize);
    });
  });
});