"use strict";
/**
 * Memory-mapped segment storage system
 * Append-only with periodic compaction for optimal performance
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.SegmentStorage = void 0;
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const tracer_js_1 = require("../telemetry/tracer.js");
const config_js_1 = require("../types/config.js");
class SegmentStorage {
    segments = new Map();
    basePath;
    constructor(basePath = './segments') {
        this.basePath = basePath;
        this.ensureDirectoryExists();
    }
    /**
     * Ensure the segments directory exists
     */
    ensureDirectoryExists() {
        if (!fs.existsSync(this.basePath)) {
            fs.mkdirSync(this.basePath, { recursive: true });
        }
    }
    /**
     * Create a new memory-mapped segment
     */
    async createSegment(segmentId, type, initialSize = 16 * 1024 * 1024 // 16MB default
    ) {
        const span = tracer_js_1.LensTracer.createChildSpan('create_segment', {
            'segment.id': segmentId,
            'segment.type': type,
            'segment.initial_size': initialSize,
        });
        try {
            const filePath = path.join(this.basePath, `${segmentId}.${type}.seg`);
            // Check if segment already exists
            if (fs.existsSync(filePath)) {
                throw new Error(`Segment ${segmentId} already exists`);
            }
            // Create file with initial size
            const fd = fs.openSync(filePath, 'w+');
            fs.ftruncateSync(fd, initialSize);
            // Create header
            const header = {
                magic: 0x4C454E53, // 'LENS'
                version: 1,
                type,
                size: initialSize,
                checksum: 0, // Will be calculated later
                created_at: Date.now(),
                last_accessed: Date.now(),
            };
            // Write header to beginning of file
            this.writeHeader(fd, header);
            // Load file into buffer (simulating mmap for extFAT compatibility)
            const buffer = Buffer.alloc(initialSize);
            // Read the header back into the buffer to keep it in sync
            fs.readSync(fd, buffer, 0, 64, 0);
            const segment = {
                file_path: filePath,
                fd,
                size: initialSize,
                buffer,
                readonly: false,
            };
            this.segments.set(segmentId, segment);
            span.setAttributes({ success: true });
            return segment;
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            span.recordException(error);
            span.setAttributes({ success: false, error: errorMsg });
            throw new Error(`Failed to create segment: ${errorMsg}`);
        }
        finally {
            span.end();
        }
    }
    /**
     * Open an existing segment for reading/writing
     */
    async openSegment(segmentId, readonly = false) {
        const span = tracer_js_1.LensTracer.createChildSpan('open_segment', {
            'segment.id': segmentId,
            'segment.readonly': readonly,
        });
        try {
            // Check if already opened
            const existing = this.segments.get(segmentId);
            if (existing) {
                // If readonly modes match, return cached segment
                if (existing.readonly === readonly) {
                    span.setAttributes({ success: true, cached: true });
                    return existing;
                }
                // If modes don't match, create a new segment instance with correct readonly mode
                // but reuse the same file descriptor and buffer
                const newSegment = {
                    ...existing,
                    readonly
                };
                this.segments.set(segmentId, newSegment);
                span.setAttributes({ success: true, cached: true, readonly_updated: true });
                return newSegment;
            }
            // Find segment file
            const files = fs.readdirSync(this.basePath);
            const segmentFile = files.find(f => f.startsWith(`${segmentId}.`) && f.endsWith('.seg'));
            if (!segmentFile) {
                throw new Error(`Segment ${segmentId} not found`);
            }
            const filePath = path.join(this.basePath, segmentFile);
            const stats = fs.statSync(filePath);
            // Open file
            const fd = fs.openSync(filePath, readonly ? 'r' : 'r+');
            // Read and validate header
            const header = this.readHeader(fd);
            this.validateHeader(header);
            // Load file into buffer (simulating mmap)
            const buffer = Buffer.alloc(stats.size);
            if (stats.size > 0) {
                fs.readSync(fd, buffer, 0, stats.size, 0);
            }
            const segment = {
                file_path: filePath,
                fd,
                size: stats.size,
                buffer,
                readonly,
            };
            this.segments.set(segmentId, segment);
            // Update access time
            if (!readonly) {
                header.last_accessed = Date.now();
                this.writeHeader(fd, header);
            }
            span.setAttributes({ success: true, cached: false });
            return segment;
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            span.recordException(error);
            span.setAttributes({ success: false, error: errorMsg });
            throw new Error(`Failed to open segment: ${errorMsg}`);
        }
        finally {
            span.end();
        }
    }
    /**
     * Write data to segment at specified offset
     */
    async writeToSegment(segmentId, offset, data) {
        const span = tracer_js_1.LensTracer.createChildSpan('write_segment', {
            'segment.id': segmentId,
            'segment.offset': offset,
            'segment.data_size': data.length,
        });
        try {
            const segment = this.segments.get(segmentId);
            if (!segment) {
                throw new Error(`Segment ${segmentId} not opened`);
            }
            if (segment.readonly) {
                throw new Error(`Segment ${segmentId} is read-only`);
            }
            // Check bounds
            if (offset + data.length > segment.size) {
                throw new Error(`Write would exceed segment size`);
            }
            // Write data to buffer and file
            data.copy(segment.buffer, offset);
            fs.writeSync(segment.fd, data, 0, data.length, offset);
            // Sync to disk
            fs.fsyncSync(segment.fd);
            span.setAttributes({ success: true });
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            span.recordException(error);
            span.setAttributes({ success: false, error: errorMsg });
            throw new Error(`Failed to write to segment: ${errorMsg}`);
        }
        finally {
            span.end();
        }
    }
    /**
     * Read data from segment at specified offset
     */
    async readFromSegment(segmentId, offset, length) {
        const span = tracer_js_1.LensTracer.createChildSpan('read_segment', {
            'segment.id': segmentId,
            'segment.offset': offset,
            'segment.length': length,
        });
        try {
            const segment = this.segments.get(segmentId);
            if (!segment) {
                throw new Error(`Segment ${segmentId} not opened`);
            }
            // Check bounds
            if (offset + length > segment.size) {
                throw new Error(`Read would exceed segment size`);
            }
            // Read data
            const data = Buffer.from(segment.buffer.subarray(offset, offset + length));
            span.setAttributes({ success: true });
            return data;
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            span.recordException(error);
            span.setAttributes({ success: false, error: errorMsg });
            throw new Error(`Failed to read from segment: ${errorMsg}`);
        }
        finally {
            span.end();
        }
    }
    /**
     * Expand segment size (append-only growth)
     */
    async expandSegment(segmentId, additionalSize) {
        const span = tracer_js_1.LensTracer.createChildSpan('expand_segment', {
            'segment.id': segmentId,
            'segment.additional_size': additionalSize,
        });
        try {
            const segment = this.segments.get(segmentId);
            if (!segment) {
                throw new Error(`Segment ${segmentId} not opened`);
            }
            if (segment.readonly) {
                throw new Error(`Segment ${segmentId} is read-only`);
            }
            const newSize = segment.size + additionalSize;
            // Check against shard size limit
            const limitMb = config_js_1.PRODUCTION_CONFIG.resources.shard_size_limit_mb;
            if (newSize > limitMb * 1024 * 1024) {
                throw new Error(`Segment would exceed size limit: ${limitMb}MB`);
            }
            // Extend file
            fs.ftruncateSync(segment.fd, newSize);
            // Create new larger buffer and copy existing data
            const newBuffer = Buffer.alloc(newSize);
            segment.buffer.copy(newBuffer);
            segment.buffer = newBuffer;
            segment.size = newSize;
            // Update header
            const header = this.readHeader(segment.fd);
            header.size = newSize;
            this.writeHeader(segment.fd, header);
            span.setAttributes({ success: true, new_size: newSize });
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            span.recordException(error);
            span.setAttributes({ success: false, error: errorMsg });
            throw new Error(`Failed to expand segment: ${errorMsg}`);
        }
        finally {
            span.end();
        }
    }
    /**
     * Close and unmap a segment
     */
    async closeSegment(segmentId) {
        const span = tracer_js_1.LensTracer.createChildSpan('close_segment', {
            'segment.id': segmentId,
        });
        try {
            const segment = this.segments.get(segmentId);
            if (!segment) {
                span.setAttributes({ success: true, already_closed: true });
                return;
            }
            // Sync buffer to disk and close
            // NOTE: Skip the header area (first 64 bytes) to avoid corrupting the header
            if (!segment.readonly && segment.buffer.length > 64) {
                fs.writeSync(segment.fd, segment.buffer, 64, segment.buffer.length - 64, 64);
                fs.fsyncSync(segment.fd);
            }
            fs.closeSync(segment.fd);
            this.segments.delete(segmentId);
            span.setAttributes({ success: true });
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            span.recordException(error);
            span.setAttributes({ success: false, error: errorMsg });
            throw new Error(`Failed to close segment: ${errorMsg}`);
        }
        finally {
            span.end();
        }
    }
    /**
     * List all segments in storage
     */
    listSegments() {
        const files = fs.readdirSync(this.basePath);
        return files
            .filter(f => f.endsWith('.seg'))
            .map(f => f.split('.')[0]);
    }
    /**
     * Get segment info without opening
     */
    async getSegmentInfo(segmentId) {
        const files = fs.readdirSync(this.basePath);
        const segmentFile = files.find(f => f.startsWith(`${segmentId}.`) && f.endsWith('.seg'));
        if (!segmentFile) {
            throw new Error(`Segment ${segmentId} not found`);
        }
        const filePath = path.join(this.basePath, segmentFile);
        const stats = fs.statSync(filePath);
        const type = segmentFile.split('.')[1];
        // Read header to get access time
        const fd = fs.openSync(filePath, 'r');
        const header = this.readHeader(fd);
        fs.closeSync(fd);
        return {
            id: segmentId,
            type,
            file_path: filePath,
            size_bytes: stats.size,
            memory_mapped: this.segments.has(segmentId),
            last_accessed: new Date(header.last_accessed),
        };
    }
    /**
     * Compact segments by removing unused space
     */
    async compactSegment(segmentId) {
        const span = tracer_js_1.LensTracer.createChildSpan('compact_segment', {
            'segment.id': segmentId,
        });
        try {
            // Implementation would analyze the segment, identify unused space,
            // create a new compacted segment, and replace the old one
            // This is a simplified placeholder
            const segment = await this.openSegment(segmentId);
            const info = await this.getSegmentInfo(segmentId);
            span.setAttributes({
                success: true,
                original_size: info.size_bytes,
                // compacted_size would be set after actual compaction
            });
            // Placeholder: actual compaction logic would go here
            console.log(`Compacted segment ${segmentId}`);
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            span.recordException(error);
            span.setAttributes({ success: false, error: errorMsg });
            throw new Error(`Failed to compact segment: ${errorMsg}`);
        }
        finally {
            span.end();
        }
    }
    /**
     * Write segment header
     */
    writeHeader(fd, header) {
        const headerBuffer = Buffer.alloc(64);
        let offset = 0;
        headerBuffer.writeUInt32LE(header.magic, offset);
        offset += 4;
        headerBuffer.writeUInt32LE(header.version, offset);
        offset += 4;
        headerBuffer.writeUInt32LE(this.typeToNumber(header.type), offset);
        offset += 4;
        headerBuffer.writeUInt32LE(header.size, offset);
        offset += 4;
        headerBuffer.writeUInt32LE(header.checksum, offset);
        offset += 4;
        headerBuffer.writeBigUInt64LE(BigInt(header.created_at), offset);
        offset += 8;
        headerBuffer.writeBigUInt64LE(BigInt(header.last_accessed), offset);
        fs.writeSync(fd, headerBuffer, 0, headerBuffer.length, 0);
    }
    /**
     * Read segment header
     */
    readHeader(fd) {
        const headerBuffer = Buffer.alloc(64);
        fs.readSync(fd, headerBuffer, 0, 64, 0);
        let offset = 0;
        const magic = headerBuffer.readUInt32LE(offset);
        offset += 4;
        const version = headerBuffer.readUInt32LE(offset);
        offset += 4;
        const type = this.numberToType(headerBuffer.readUInt32LE(offset));
        offset += 4;
        const size = headerBuffer.readUInt32LE(offset);
        offset += 4;
        const checksum = headerBuffer.readUInt32LE(offset);
        offset += 4;
        const created_at = Number(headerBuffer.readBigUInt64LE(offset));
        offset += 8;
        const last_accessed = Number(headerBuffer.readBigUInt64LE(offset));
        return { magic, version, type, size, checksum, created_at, last_accessed };
    }
    /**
     * Validate segment header
     */
    validateHeader(header) {
        if (header.magic !== 0x4C454E53) {
            throw new Error('Invalid segment magic number');
        }
        if (header.version !== 1) {
            throw new Error(`Unsupported segment version: ${header.version}`);
        }
    }
    /**
     * Convert segment type to number for storage
     */
    typeToNumber(type) {
        const types = { lexical: 1, symbols: 2, ast: 3, semantic: 4 };
        return types[type] || 0;
    }
    /**
     * Convert number to segment type
     */
    numberToType(num) {
        const types = { 1: 'lexical', 2: 'symbols', 3: 'ast', 4: 'semantic' };
        return (types[num] || 'lexical');
    }
    /**
     * Cleanup - close all segments
     */
    async shutdown() {
        const segmentIds = Array.from(this.segments.keys());
        await Promise.all(segmentIds.map(id => this.closeSegment(id)));
        console.log('All segments closed');
    }
}
exports.SegmentStorage = SegmentStorage;
