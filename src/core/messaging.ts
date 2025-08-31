/**
 * NATS/JetStream messaging system for work distribution
 * Handles work units across ingest, query, and maintenance pools
 */

import { connect, NatsConnection, JetStreamManager, JetStreamClient, StringCodec } from 'nats';
import type { WorkUnit, WorkType } from '../types/core.js';
import { LensTracer } from '../telemetry/tracer.js';
import { PRODUCTION_CONFIG } from '../types/config.js';

const sc = StringCodec();

export class MessagingSystem {
  private nc?: NatsConnection;
  private jsm?: JetStreamManager;
  private js?: JetStreamClient;
  private isConnected = false;

  constructor(
    private natsUrl: string = 'nats://localhost:4222',
    private streamName: string = 'LENS_WORK'
  ) {}

  /**
   * Initialize NATS connection and JetStream
   */
  async initialize(): Promise<void> {
    const span = LensTracer.createChildSpan('messaging_init');
    
    try {
      // Connect to NATS
      this.nc = await connect({ 
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
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Failed to initialize messaging system: ${errorMsg}`);
    } finally {
      span.end();
    }
  }

  /**
   * Create JetStream work stream with appropriate configuration
   */
  private async createWorkStream(): Promise<void> {
    if (!this.jsm) {
      throw new Error('JetStream manager not initialized');
    }

    try {
      await this.jsm.streams.info(this.streamName);
      console.log(`Stream ${this.streamName} already exists`);
    } catch {
      // Stream doesn't exist, create it
      await this.jsm.streams.add({
        name: this.streamName,
        subjects: ['lens.work.*'],
        max_msgs: 10000,
        max_bytes: 100 * 1024 * 1024, // 100MB
        max_age: 24 * 60 * 60 * 1000_000_000, // 24 hours in nanoseconds
        storage: 'file' as any,
        retention: 'workqueue' as any,
        discard: 'old' as any,
      });
      
      console.log(`Created JetStream stream: ${this.streamName}`);
    }
  }

  /**
   * Publish a work unit to the appropriate subject
   */
  async publishWork(workUnit: WorkUnit): Promise<void> {
    if (!this.js || !this.isConnected) {
      throw new Error('Messaging system not connected');
    }

    const span = LensTracer.createChildSpan('publish_work', {
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
        } as any,
      });

      span.setAttributes({ success: true });
      
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      span.recordException(error as Error);
      span.setAttributes({ success: false, error: errorMsg });
      throw new Error(`Failed to publish work unit: ${errorMsg}`);
    } finally {
      span.end();
    }
  }

  /**
   * Subscribe to work units for a specific worker pool
   */
  async subscribeToWork(
    workType: WorkType,
    handler: (workUnit: WorkUnit) => Promise<void>,
    concurrency: number = 1
  ): Promise<void> {
    if (!this.js || !this.isConnected) {
      throw new Error('Messaging system not connected');
    }

    const subject = `lens.work.${workType}`;
    const consumer = await (this.js.consumers as any).add(this.streamName, {
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
      const span = LensTracer.createChildSpan('process_work', {
        'work.type': workType,
        'work.subject': msg.subject,
      });

      try {
        const workUnit: WorkUnit = JSON.parse(sc.decode(msg.data));
        
        span.setAttributes({
          'work.id': workUnit.id,
          'work.shard_id': workUnit.shard_id,
          'work.priority': workUnit.priority,
        });

        await handler(workUnit);
        msg.ack();
        
        span.setAttributes({ success: true });
        
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error';
        span.recordException(error as Error);
        span.setAttributes({ success: false, error: errorMsg });
        
        // Negative ack to retry
        msg.nak(1000); // 1 second delay
        
      } finally {
        span.end();
      }
    }
  }

  /**
   * Create a work unit for indexing a shard
   */
  static createIndexWork(repoSha: string, shardId: string): WorkUnit {
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
  static createCompactionWork(repoSha: string, shardId: string): WorkUnit {
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
  static createSymbolWork(repoSha: string, shardId: string, filePaths: string[]): WorkUnit {
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
      } else if (consumer.name.includes('query')) {
        workerStatus.query_active += consumer.num_pending || 0;
      } else if (consumer.name.includes('compact') || consumer.name.includes('health')) {
        workerStatus.maintenance_active += consumer.num_pending || 0;
      }
    }

    return workerStatus;
  }

  /**
   * Graceful shutdown
   */
  async shutdown(): Promise<void> {
    if (this.nc && this.isConnected) {
      await this.nc.drain();
      await this.nc.close();
      this.isConnected = false;
      console.log('NATS connection closed');
    }
  }
}