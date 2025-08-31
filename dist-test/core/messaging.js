"use strict";
/**
 * NATS/JetStream messaging system for work distribution
 * Handles work units across ingest, query, and maintenance pools
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.MessagingSystem = void 0;
const nats_1 = require("nats");
const tracer_js_1 = require("../telemetry/tracer.js");
const sc = (0, nats_1.StringCodec)();
class MessagingSystem {
    natsUrl;
    streamName;
    nc;
    jsm;
    js;
    isConnected = false;
    constructor(natsUrl = 'nats://localhost:4222', streamName = 'LENS_WORK') {
        this.natsUrl = natsUrl;
        this.streamName = streamName;
    }
    /**
     * Initialize NATS connection and JetStream
     */
    async initialize() {
        const span = tracer_js_1.LensTracer.createChildSpan('messaging_init');
        try {
            // Connect to NATS
            this.nc = await (0, nats_1.connect)({
                servers: this.natsUrl,
                reconnect: true,
                maxReconnectAttempts: 10,
                reconnectTimeWait: 1000,
            });
            console.log('Connected to NATS server');
            // Get JetStream manager and client
            this.jsm = await this.nc.jetstreamManager();
            this.js = this.nc.jetstream();
            // Create the work stream if it doesn't exist
            await this.createWorkStream();
            this.isConnected = true;
            span.setAttributes({ success: true });
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            span.recordException(error);
            span.setAttributes({ success: false, error: errorMsg });
            throw new Error(`Failed to initialize messaging system: ${errorMsg}`);
        }
        finally {
            span.end();
        }
    }
    /**
     * Create JetStream work stream with appropriate configuration
     */
    async createWorkStream() {
        if (!this.jsm) {
            throw new Error('JetStream manager not initialized');
        }
        try {
            await this.jsm.streams.info(this.streamName);
            console.log(`Stream ${this.streamName} already exists`);
        }
        catch {
            // Stream doesn't exist, create it
            await this.jsm.streams.add({
                name: this.streamName,
                subjects: ['lens.work.*'],
                max_msgs: 10000,
                max_bytes: 100 * 1024 * 1024, // 100MB
                max_age: 24 * 60 * 60 * 1000_000_000, // 24 hours in nanoseconds
                storage: 'file',
                retention: 'workqueue',
                discard: 'old',
            });
            console.log(`Created JetStream stream: ${this.streamName}`);
        }
    }
    /**
     * Publish a work unit to the appropriate subject
     */
    async publishWork(workUnit) {
        if (!this.js || !this.isConnected) {
            throw new Error('Messaging system not connected');
        }
        const span = tracer_js_1.LensTracer.createChildSpan('publish_work', {
            'work.id': workUnit.id,
            'work.type': workUnit.type,
            'work.shard_id': workUnit.shard_id,
            'work.priority': workUnit.priority,
        });
        try {
            const subject = `lens.work.${workUnit.type}`;
            const payload = JSON.stringify(workUnit);
            await this.js.publish(subject, sc.encode(payload), {
                headers: {
                    'Lens-Work-ID': workUnit.id,
                    'Lens-Work-Type': workUnit.type,
                    'Lens-Priority': workUnit.priority.toString(),
                },
            });
            span.setAttributes({ success: true });
        }
        catch (error) {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            span.recordException(error);
            span.setAttributes({ success: false, error: errorMsg });
            throw new Error(`Failed to publish work unit: ${errorMsg}`);
        }
        finally {
            span.end();
        }
    }
    /**
     * Subscribe to work units for a specific worker pool
     */
    async subscribeToWork(workType, handler, concurrency = 1) {
        if (!this.js || !this.isConnected) {
            throw new Error('Messaging system not connected');
        }
        const subject = `lens.work.${workType}`;
        const consumer = await this.js.consumers.add(this.streamName, {
            durable_name: `lens_${workType}_worker`,
            filter_subject: subject,
            ack_policy: 'explicit',
            max_deliver: 3,
            ack_wait: 30_000_000_000, // 30 seconds in nanoseconds
            max_ack_pending: concurrency,
        });
        const messages = await consumer.consume({
            max_messages: concurrency,
        });
        // Process messages concurrently
        for await (const msg of messages) {
            const span = tracer_js_1.LensTracer.createChildSpan('process_work', {
                'work.type': workType,
                'work.subject': msg.subject,
            });
            try {
                const workUnit = JSON.parse(sc.decode(msg.data));
                span.setAttributes({
                    'work.id': workUnit.id,
                    'work.shard_id': workUnit.shard_id,
                    'work.priority': workUnit.priority,
                });
                await handler(workUnit);
                msg.ack();
                span.setAttributes({ success: true });
            }
            catch (error) {
                const errorMsg = error instanceof Error ? error.message : 'Unknown error';
                span.recordException(error);
                span.setAttributes({ success: false, error: errorMsg });
                // Negative ack to retry
                msg.nak(1000); // 1 second delay
            }
            finally {
                span.end();
            }
        }
    }
    /**
     * Create a work unit for indexing a shard
     */
    static createIndexWork(repoSha, shardId) {
        return {
            id: `index_${shardId}_${Date.now()}`,
            type: 'index_shard',
            repo_sha: repoSha,
            shard_id: shardId,
            payload: { shard_id: shardId },
            priority: 5, // Normal priority
            created_at: new Date(),
        };
    }
    /**
     * Create a work unit for compacting a shard
     */
    static createCompactionWork(repoSha, shardId) {
        return {
            id: `compact_${shardId}_${Date.now()}`,
            type: 'compact_shard',
            repo_sha: repoSha,
            shard_id: shardId,
            payload: { shard_id: shardId },
            priority: 3, // Lower priority
            created_at: new Date(),
        };
    }
    /**
     * Create a work unit for building symbols
     */
    static createSymbolWork(repoSha, shardId, filePaths) {
        return {
            id: `symbols_${shardId}_${Date.now()}`,
            type: 'build_symbols',
            repo_sha: repoSha,
            shard_id: shardId,
            payload: { shard_id: shardId, file_paths: filePaths },
            priority: 7, // Higher priority
            created_at: new Date(),
        };
    }
    /**
     * Get stream information
     */
    async getStreamInfo() {
        if (!this.jsm) {
            throw new Error('JetStream manager not initialized');
        }
        return await this.jsm.streams.info(this.streamName);
    }
    /**
     * Get worker pool status
     */
    async getWorkerStatus() {
        if (!this.jsm) {
            throw new Error('JetStream manager not initialized');
        }
        const consumers = await this.jsm.consumers.list(this.streamName).next();
        const workerStatus = {
            ingest_active: 0,
            query_active: 0,
            maintenance_active: 0,
        };
        for (const consumer of consumers) {
            if (consumer.name.includes('index_shard') || consumer.name.includes('build_')) {
                workerStatus.ingest_active += consumer.num_pending || 0;
            }
            else if (consumer.name.includes('query')) {
                workerStatus.query_active += consumer.num_pending || 0;
            }
            else if (consumer.name.includes('compact') || consumer.name.includes('health')) {
                workerStatus.maintenance_active += consumer.num_pending || 0;
            }
        }
        return workerStatus;
    }
    /**
     * Graceful shutdown
     */
    async shutdown() {
        if (this.nc && this.isConnected) {
            await this.nc.drain();
            await this.nc.close();
            this.isConnected = false;
            console.log('NATS connection closed');
        }
    }
}
exports.MessagingSystem = MessagingSystem;
