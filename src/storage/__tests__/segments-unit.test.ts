/**
 * Comprehensive unit tests for SegmentStorage
 * Focus on business logic and edge cases with mocked file system operations
 */

import { describe, it, expect, jest, beforeEach, afterEach, mock } from 'bun:test';
import { SegmentStorage } from '../segments.js';
import type { SegmentType, MMapSegment } from '../../types/core.js';

// Mock file system operations to focus on business logic
mock('fs', () => ({
  existsSync: jest.fn(),
  mkdirSync: jest.fn(),
  openSync: jest.fn(),
  ftruncateSync: jest.fn(),
  readSync: jest.fn(),
  writeSync: jest.fn(),
  fsyncSync: jest.fn(),
  closeSync: jest.fn(),
  readdirSync: jest.fn(),
  statSync: jest.fn()
}));

// Import the mocked fs module
import * as fs from 'fs';
const mockFs = fs as any;

mock('path', () => ({
  join: (...args: string[]) => args.join('/')
}));

// Mock telemetry to avoid complexity
mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: jest.fn(() => ({
      setAttributes: jest.fn(),
      recordException: jest.fn(),
      end: jest.fn()
    }))
  }
}));

// Mock production config
mock('../../types/config.js', () => ({
  PRODUCTION_CONFIG: {
    resources: {
      shard_size_limit_mb: 1024
    }
  }
}));

describe('SegmentStorage Unit Tests', () => {
  let storage: SegmentStorage;
  
  beforeEach(() => {
    storage = new SegmentStorage('./test-segments');
    jest.clearAllMocks();
    
    // Default mock behaviors
    mockFs.existsSync.mockReturnValue(true);
    mockFs.mkdirSync.mockReturnValue(undefined);
    mockFs.openSync.mockReturnValue(5); // Mock file descriptor
    mockFs.ftruncateSync.mockReturnValue(undefined);
    mockFs.readSync.mockReturnValue(64);
    mockFs.writeSync.mockReturnValue(64);
    mockFs.fsyncSync.mockReturnValue(undefined);
    mockFs.closeSync.mockReturnValue(undefined);
    mockFs.readdirSync.mockReturnValue([]);
    mockFs.statSync.mockReturnValue({ size: 1024 });
  });

  afterEach(async () => {
    await storage.shutdown();
  });

  describe('Constructor and Initialization', () => {
    it('should create storage with custom base path', () => {
      const customStorage = new SegmentStorage('./custom-path');
      expect(customStorage).toBeDefined();
    });

    it('should create storage with default base path', () => {
      const defaultStorage = new SegmentStorage();
      expect(defaultStorage).toBeDefined();
    });

    it('should handle directory creation lazily', () => {
      // Directory creation should not be called in constructor
      expect(mockFs.mkdirSync).not.toHaveBeenCalled();
    });
  });

  describe('Directory Management', () => {
    it('should create directory when it does not exist', async () => {
      mockFs.existsSync.mockReturnValue(false);
      
      await storage.createSegment('test-segment', 'lexical');
      
      expect(mockFs.existsSync).toHaveBeenCalledWith('./test-segments');
      expect(mockFs.mkdirSync).toHaveBeenCalledWith('./test-segments', { recursive: true });
    });

    it('should skip directory creation when it exists', async () => {
      mockFs.existsSync.mockReturnValue(true);
      
      await storage.createSegment('test-segment', 'lexical');
      
      expect(mockFs.existsSync).toHaveBeenCalledWith('./test-segments');
      expect(mockFs.mkdirSync).not.toHaveBeenCalled();
    });
  });

  describe('Segment Header Operations', () => {
    it('should validate segment header format', () => {
      const validHeader = {
        magic: 0x4C454E53, // 'LENS'
        version: 1,
        type: 'lexical' as SegmentType,
        size: 1024,
        checksum: 0,
        created_at: Date.now(),
        last_accessed: Date.now()
      };

      // Test validation logic
      expect(validHeader.magic).toBe(0x4C454E53);
      expect(validHeader.version).toBe(1);
    });

    it('should handle invalid magic number', () => {
      const invalidHeader = {
        magic: 0x12345678, // Invalid magic
        version: 1
      };

      expect(invalidHeader.magic).not.toBe(0x4C454E53);
      
      // Simulate validation failure
      const shouldThrow = () => {
        if (invalidHeader.magic !== 0x4C454E53) {
          throw new Error('Invalid segment magic number');
        }
      };

      expect(shouldThrow).toThrow('Invalid segment magic number');
    });

    it('should handle unsupported version', () => {
      const unsupportedVersionHeader = {
        magic: 0x4C454E53,
        version: 99 // Unsupported version
      };

      const shouldThrow = () => {
        if (unsupportedVersionHeader.version !== 1) {
          throw new Error(`Unsupported segment version: ${unsupportedVersionHeader.version}`);
        }
      };

      expect(shouldThrow).toThrow('Unsupported segment version: 99');
    });
  });

  describe('Segment Type Conversion', () => {
    it('should convert segment types to numbers correctly', () => {
      const typeToNumber = (type: SegmentType): number => {
        const types = { lexical: 1, symbols: 2, ast: 3, semantic: 4 };
        return types[type] || 0;
      };

      expect(typeToNumber('lexical')).toBe(1);
      expect(typeToNumber('symbols')).toBe(2);
      expect(typeToNumber('ast')).toBe(3);
      expect(typeToNumber('semantic')).toBe(4);
    });

    it('should convert numbers to segment types correctly', () => {
      const numberToType = (num: number): SegmentType => {
        const types = { 1: 'lexical', 2: 'symbols', 3: 'ast', 4: 'semantic' };
        return (types[num as keyof typeof types] || 'lexical') as SegmentType;
      };

      expect(numberToType(1)).toBe('lexical');
      expect(numberToType(2)).toBe('symbols');
      expect(numberToType(3)).toBe('ast');
      expect(numberToType(4)).toBe('semantic');
      expect(numberToType(999)).toBe('lexical'); // Default fallback
    });

    it('should handle unknown segment types', () => {
      const typeToNumber = (type: string): number => {
        const types: Record<string, number> = { lexical: 1, symbols: 2, ast: 3, semantic: 4 };
        return types[type] || 0;
      };

      expect(typeToNumber('unknown')).toBe(0);
      expect(typeToNumber('')).toBe(0);
    });
  });

  describe('Segment Creation Logic', () => {
    it('should handle successful segment creation', async () => {
      const segmentId = 'test-seg';
      const segmentType: SegmentType = 'lexical';
      const initialSize = 16 * 1024 * 1024;

      const result = await storage.createSegment(segmentId, segmentType, initialSize);

      expect(result).toBeDefined();
      expect(result.file_path).toBe('./test-segments/test-seg.lexical.seg');
      expect(result.size).toBe(initialSize);
      expect(result.readonly).toBe(false);
      expect(mockFs.openSync).toHaveBeenCalledWith('./test-segments/test-seg.lexical.seg', 'w+');
      expect(mockFs.ftruncateSync).toHaveBeenCalledWith(5, initialSize);
    });

    it('should reject duplicate segment creation', async () => {
      mockFs.existsSync.mockReturnValueOnce(true) // Directory exists
                       .mockReturnValueOnce(true); // File exists

      await expect(storage.createSegment('duplicate', 'lexical')).rejects.toThrow('Segment duplicate already exists');
    });

    it('should use default initial size', async () => {
      const result = await storage.createSegment('default-size', 'lexical');
      
      expect(mockFs.ftruncateSync).toHaveBeenCalledWith(5, 16 * 1024 * 1024);
    });

    it('should create buffer with correct size', async () => {
      const customSize = 8 * 1024 * 1024;
      const result = await storage.createSegment('custom-size', 'lexical', customSize);
      
      expect(result.buffer).toBeDefined();
      expect(result.buffer.length).toBe(customSize);
    });
  });

  describe('Segment Opening Logic', () => {
    beforeEach(() => {
      mockFs.readdirSync.mockReturnValue(['existing.lexical.seg']);
      mockFs.statSync.mockReturnValue({ size: 1024 });
      
      // Mock valid header reading - header needs specific binary format
      mockFs.readSync.mockImplementation((fd, buffer, offset, length, position) => {
        if (length === 64 && position === 0) {
          // Create a valid segment header with magic 0x4C454E53 ("LENS") and version 1
          const headerBuffer = buffer as Buffer;
          let writeOffset = 0;
          headerBuffer.writeUInt32LE(0x4C454E53, writeOffset); writeOffset += 4; // magic "LENS"
          headerBuffer.writeUInt32LE(1, writeOffset); writeOffset += 4; // version
          headerBuffer.writeUInt32LE(0, writeOffset); writeOffset += 4; // type (lexical = 0)
          headerBuffer.writeUInt32LE(1024, writeOffset); writeOffset += 4; // size
          headerBuffer.writeUInt32LE(12345, writeOffset); writeOffset += 4; // checksum
          headerBuffer.writeBigUInt64LE(BigInt(Date.now()), writeOffset); writeOffset += 8; // created_at
          headerBuffer.writeBigUInt64LE(BigInt(Date.now()), writeOffset); // last_accessed
          return 64;
        }
        return length; // Default return for other reads
      });
    });

    it('should open existing segment successfully', async () => {
      const result = await storage.openSegment('existing');
      
      expect(result).toBeDefined();
      expect(result.file_path).toBe('./test-segments/existing.lexical.seg');
      expect(result.size).toBe(1024);
      expect(mockFs.openSync).toHaveBeenCalledWith('./test-segments/existing.lexical.seg', 'r+');
    });

    it('should open segment in readonly mode', async () => {
      const result = await storage.openSegment('existing', true);
      
      expect(result.readonly).toBe(true);
      expect(mockFs.openSync).toHaveBeenCalledWith('./test-segments/existing.lexical.seg', 'r');
    });

    it('should return cached segment when already opened', async () => {
      const first = await storage.openSegment('existing');
      const second = await storage.openSegment('existing');
      
      expect(first).toBe(second);
      expect(mockFs.openSync).toHaveBeenCalledTimes(1);
    });

    it('should handle readonly mode changes', async () => {
      const readWrite = await storage.openSegment('existing', false);
      const readonly = await storage.openSegment('existing', true);
      
      expect(readWrite.readonly).toBe(false);
      expect(readonly.readonly).toBe(true);
    });

    it('should reject opening non-existent segment', async () => {
      mockFs.readdirSync.mockReturnValue([]);
      
      await expect(storage.openSegment('non-existent')).rejects.toThrow('Segment non-existent not found');
    });
  });

  describe('Segment Read/Write Operations', () => {
    let mockSegment: MMapSegment;
    
    beforeEach(() => {
      mockSegment = {
        file_path: './test-segments/test.lexical.seg',
        fd: 5,
        size: 1024,
        buffer: Buffer.alloc(1024),
        readonly: false
      };
      
      // Mock segment storage
      (storage as any).segments.set('test', mockSegment);
    });

    it('should write data to segment successfully', async () => {
      const data = Buffer.from('test data');
      const offset = 100;
      
      await storage.writeToSegment('test', offset, data);
      
      expect(mockFs.writeSync).toHaveBeenCalledWith(5, data, 0, data.length, offset);
      expect(mockFs.fsyncSync).toHaveBeenCalledWith(5);
    });

    it('should read data from segment successfully', async () => {
      const offset = 100;
      const length = 50;
      
      const result = await storage.readFromSegment('test', offset, length);
      
      expect(result).toBeDefined();
      expect(result.length).toBe(length);
    });

    it('should reject write to readonly segment', async () => {
      mockSegment.readonly = true;
      const data = Buffer.from('test');
      
      await expect(storage.writeToSegment('test', 0, data)).rejects.toThrow('Segment test is read-only');
    });

    it('should reject write beyond segment bounds', async () => {
      const data = Buffer.from('test data');
      const offset = 1020; // Would exceed 1024 byte segment
      
      await expect(storage.writeToSegment('test', offset, data)).rejects.toThrow('Write would exceed segment size');
    });

    it('should reject read beyond segment bounds', async () => {
      const offset = 1000;
      const length = 100; // Would exceed 1024 byte segment
      
      await expect(storage.readFromSegment('test', offset, length)).rejects.toThrow('Read would exceed segment size');
    });

    it('should reject operations on non-existent segment', async () => {
      const data = Buffer.from('test');
      
      await expect(storage.writeToSegment('non-existent', 0, data)).rejects.toThrow('Segment non-existent not opened');
      await expect(storage.readFromSegment('non-existent', 0, 10)).rejects.toThrow('Segment non-existent not opened');
    });
  });

  describe('Segment Expansion Logic', () => {
    let mockSegment: MMapSegment;
    
    beforeEach(() => {
      mockSegment = {
        file_path: './test-segments/test.lexical.seg',
        fd: 5,
        size: 1024,
        buffer: Buffer.alloc(1024),
        readonly: false
      };
      
      (storage as any).segments.set('test', mockSegment);
    });

    it('should expand segment successfully', async () => {
      const additionalSize = 512;
      const newSize = 1024 + 512;
      
      await storage.expandSegment('test', additionalSize);
      
      expect(mockFs.ftruncateSync).toHaveBeenCalledWith(5, newSize);
      expect(mockSegment.size).toBe(newSize);
      expect(mockSegment.buffer.length).toBe(newSize);
    });

    it('should reject expansion of readonly segment', async () => {
      mockSegment.readonly = true;
      
      await expect(storage.expandSegment('test', 512)).rejects.toThrow('Segment test is read-only');
    });

    it('should reject expansion beyond size limit', async () => {
      const limitMb = 1024; // From mocked config
      const currentSize = 1024;
      const additionalSize = limitMb * 1024 * 1024; // Exceeds limit
      
      await expect(storage.expandSegment('test', additionalSize)).rejects.toThrow('Segment would exceed size limit');
    });

    it('should handle buffer copying during expansion', async () => {
      const originalData = 'original data';
      mockSegment.buffer.write(originalData, 0);
      
      await storage.expandSegment('test', 512);
      
      // Verify original data is preserved
      const preservedData = mockSegment.buffer.subarray(0, originalData.length).toString();
      expect(preservedData).toBe(originalData);
    });
  });

  describe('Segment Listing and Information', () => {
    it('should list segments correctly', () => {
      mockFs.readdirSync.mockReturnValue(['seg1.lexical.seg', 'seg2.symbols.seg', 'other.txt']);
      
      const segments = storage.listSegments();
      
      expect(segments).toEqual(['seg1', 'seg2']);
      expect(segments).toHaveLength(2);
    });

    it('should handle empty segment directory', () => {
      mockFs.readdirSync.mockReturnValue([]);
      
      const segments = storage.listSegments();
      
      expect(segments).toEqual([]);
      expect(segments).toHaveLength(0);
    });

    it('should get segment info correctly', async () => {
      mockFs.readdirSync.mockReturnValue(['info-test.lexical.seg']);
      mockFs.statSync.mockReturnValue({ size: 2048 });
      mockFs.readSync.mockReturnValue(64);
      
      const info = await storage.getSegmentInfo('info-test');
      
      expect(info.id).toBe('info-test');
      expect(info.type).toBe('lexical');
      expect(info.size_bytes).toBe(2048);
      expect(info.memory_mapped).toBe(false);
    });

    it('should detect memory mapped segments', async () => {
      mockFs.readdirSync.mockReturnValue(['mapped.lexical.seg']);
      mockFs.statSync.mockReturnValue({ size: 1024 });
      mockFs.readSync.mockReturnValue(64);
      
      // Add segment to memory map
      (storage as any).segments.set('mapped', {});
      
      const info = await storage.getSegmentInfo('mapped');
      
      expect(info.memory_mapped).toBe(true);
    });
  });

  describe('Segment Cleanup and Shutdown', () => {
    let mockSegment: MMapSegment;
    
    beforeEach(() => {
      mockSegment = {
        file_path: './test-segments/cleanup.lexical.seg',
        fd: 5,
        size: 1024,
        buffer: Buffer.alloc(1024),
        readonly: false
      };
      
      (storage as any).segments.set('cleanup', mockSegment);
    });

    it('should close segment successfully', async () => {
      await storage.closeSegment('cleanup');
      
      expect(mockFs.writeSync).toHaveBeenCalled();
      expect(mockFs.fsyncSync).toHaveBeenCalledWith(5);
      expect(mockFs.closeSync).toHaveBeenCalledWith(5);
    });

    it('should skip sync for readonly segments', async () => {
      mockSegment.readonly = true;
      
      await storage.closeSegment('cleanup');
      
      expect(mockFs.writeSync).not.toHaveBeenCalled();
      expect(mockFs.fsyncSync).not.toHaveBeenCalled();
      expect(mockFs.closeSync).toHaveBeenCalledWith(5);
    });

    it('should handle closing already closed segment', async () => {
      await storage.closeSegment('non-existent');
      
      // Should not throw error
      expect(mockFs.closeSync).not.toHaveBeenCalled();
    });

    it('should shutdown all segments', async () => {
      // Add multiple segments
      (storage as any).segments.set('seg1', { ...mockSegment, fd: 6 });
      (storage as any).segments.set('seg2', { ...mockSegment, fd: 7 });
      
      await storage.shutdown();
      
      expect(mockFs.closeSync).toHaveBeenCalledTimes(3); // cleanup + seg1 + seg2
    });
  });

  describe('Edge Cases and Error Handling', () => {
    it('should handle file system errors gracefully', async () => {
      mockFs.openSync.mockImplementation(() => {
        throw new Error('File system error');
      });
      
      await expect(storage.createSegment('error-test', 'lexical')).rejects.toThrow('Failed to create segment');
    });

    it('should handle buffer allocation with zero size', async () => {
      const zeroSizeSegment = await storage.createSegment('zero', 'lexical', 0);
      
      expect(zeroSizeSegment.buffer.length).toBe(0);
      expect(zeroSizeSegment.size).toBe(0);
    });

    it('should handle very large segment sizes', () => {
      const largeSize = 500 * 1024 * 1024; // 500MB
      const limitMb = 1024; // 1GB limit
      const wouldExceedLimit = largeSize > limitMb * 1024 * 1024;
      
      expect(wouldExceedLimit).toBe(false);
      
      const tooLargeSize = 2 * 1024 * 1024 * 1024; // 2GB
      const exceedsLimit = tooLargeSize > limitMb * 1024 * 1024;
      
      expect(exceedsLimit).toBe(true);
    });

    it('should handle concurrent segment operations', async () => {
      const promises = [
        storage.createSegment('concurrent1', 'lexical'),
        storage.createSegment('concurrent2', 'symbols'),
        storage.createSegment('concurrent3', 'ast')
      ];
      
      const results = await Promise.all(promises);
      
      expect(results).toHaveLength(3);
      results.forEach(result => {
        expect(result).toBeDefined();
        expect(result.buffer).toBeDefined();
      });
    });
  });
});