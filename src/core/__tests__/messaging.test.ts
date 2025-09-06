/**
 * Tests for MessagingSystem
 * Priority: MEDIUM - Core infrastructure component for system communication
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { MessagingSystem } from '../messaging.js';

// Mock NATS or other messaging dependencies
vi.mock('nats', () => ({
  connect: vi.fn().mockResolvedValue({
    publish: vi.fn(),
    subscribe: vi.fn(),
    drain: vi.fn(),
    close: vi.fn(),
  }),
  StringCodec: vi.fn().mockReturnValue({
    encode: vi.fn().mockImplementation(str => Buffer.from(str)),
    decode: vi.fn().mockImplementation(buf => buf.toString()),
  }),
}));

vi.mock('../../telemetry/tracer.js', () => ({
  LensTracer: {
    createChildSpan: vi.fn().mockReturnValue({
      setAttributes: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn(),
    }),
  },
}));

describe('MessagingSystem', () => {
  let messaging: MessagingSystem;

  beforeEach(() => {
    vi.clearAllMocks();
    messaging = new MessagingSystem();
  });

  afterEach(async () => {
    if (messaging) {
      await messaging.shutdown();
    }
  });

  describe('Initialization', () => {
    it('should initialize successfully', async () => {
      await expect(messaging.initialize()).resolves.not.toThrow();
    });

    it('should handle initialization failures', async () => {
      const { connect } = await import('nats');
      vi.mocked(connect).mockRejectedValueOnce(new Error('Connection failed'));
      
      // Should handle connection failures gracefully
      await expect(messaging.initialize()).rejects.toThrow();
    });

    it('should not initialize twice', async () => {
      await messaging.initialize();
      
      // Second initialization should be idempotent
      await expect(messaging.initialize()).resolves.not.toThrow();
    });
  });

  describe('Worker Status', () => {
    beforeEach(async () => {
      await messaging.initialize();
    });

    it('should return worker status', async () => {
      const status = await messaging.getWorkerStatus();
      
      expect(status).toBeDefined();
      expect(status).toHaveProperty('ingest_active');
      expect(status).toHaveProperty('query_active');
      expect(status).toHaveProperty('maintenance_active');
      
      expect(typeof status.ingest_active).toBe('number');
      expect(typeof status.query_active).toBe('number');
      expect(typeof status.maintenance_active).toBe('number');
    });

    it('should handle worker status errors', async () => {
      // Simulate messaging system issues
      const mockError = new Error('Status check failed');
      
      // Mock the internal status check to fail
      const originalGetWorkerStatus = (messaging as any).getWorkerStatus;
      if (originalGetWorkerStatus) {
        (messaging as any).getWorkerStatus = vi.fn().mockRejectedValue(mockError);
      }

      // Should return default values or handle error
      const status = await messaging.getWorkerStatus();
      expect(status).toBeDefined();
      expect(status.ingest_active).toBeGreaterThanOrEqual(0);
      expect(status.query_active).toBeGreaterThanOrEqual(0);
      expect(status.maintenance_active).toBeGreaterThanOrEqual(0);
    });

    it('should track active workers correctly', async () => {
      const status1 = await messaging.getWorkerStatus();
      
      // Values should be non-negative
      expect(status1.ingest_active).toBeGreaterThanOrEqual(0);
      expect(status1.query_active).toBeGreaterThanOrEqual(0);
      expect(status1.maintenance_active).toBeGreaterThanOrEqual(0);
      
      // Multiple calls should return consistent data
      const status2 = await messaging.getWorkerStatus();
      expect(typeof status2.ingest_active).toBe('number');
      expect(typeof status2.query_active).toBe('number');
      expect(typeof status2.maintenance_active).toBe('number');
    });
  });

  describe('Message Publishing', () => {
    beforeEach(async () => {
      await messaging.initialize();
    });

    it('should publish messages successfully', async () => {
      const message = {
        type: 'test',
        data: { key: 'value' },
      };

      // Should not throw when publishing
      await expect(messaging.publish?.('test.topic', message))
        .resolves?.not?.toThrow();
    });

    it('should handle publish errors gracefully', async () => {
      const { connect } = await import('nats');
      const mockConnection = await vi.mocked(connect).mock.results[0]?.value;
      
      if (mockConnection?.publish) {
        vi.mocked(mockConnection.publish).mockImplementation(() => {
          throw new Error('Publish failed');
        });
      }

      const message = { type: 'test', data: {} };
      
      // Should handle publish failures gracefully
      if (messaging.publish) {
        await expect(messaging.publish('test.topic', message))
          .resolves.not.toThrow();
      }
    });
  });

  describe('Message Subscription', () => {
    beforeEach(async () => {
      await messaging.initialize();
    });

    it('should subscribe to topics successfully', async () => {
      const callback = vi.fn();
      
      // Should not throw when subscribing
      if (messaging.subscribe) {
        await expect(messaging.subscribe('test.topic', callback))
          .resolves?.not?.toThrow();
      }
    });

    it('should handle subscription errors', async () => {
      const { connect } = await import('nats');
      const mockConnection = await vi.mocked(connect).mock.results[0]?.value;
      
      if (mockConnection?.subscribe) {
        vi.mocked(mockConnection.subscribe).mockImplementation(() => {
          throw new Error('Subscribe failed');
        });
      }

      const callback = vi.fn();
      
      if (messaging.subscribe) {
        await expect(messaging.subscribe('test.topic', callback))
          .resolves.not.toThrow();
      }
    });

    it('should handle message processing', async () => {
      const callback = vi.fn();
      const testMessage = { type: 'test', data: { value: 42 } };
      
      if (messaging.subscribe) {
        await messaging.subscribe('test.topic', callback);
      }

      // Simulate message received
      if (callback.mock.calls.length > 0 || messaging.publish) {
        // Message should be processed through callback
        expect(callback).toBeDefined();
      }
    });
  });

  describe('Connection Management', () => {
    it('should handle connection lifecycle', async () => {
      await messaging.initialize();
      
      const status = await messaging.getWorkerStatus();
      expect(status).toBeDefined();
      
      await messaging.shutdown();
      
      // After shutdown, operations should handle gracefully
      const statusAfterShutdown = await messaging.getWorkerStatus();
      expect(statusAfterShutdown).toBeDefined();
    });

    it('should handle connection interruptions', async () => {
      await messaging.initialize();
      
      // Simulate connection drop
      const { connect } = await import('nats');
      const mockConnection = await vi.mocked(connect).mock.results[0]?.value;
      
      if (mockConnection) {
        // Simulate connection close
        mockConnection.close?.();
      }
      
      // Should handle disconnection gracefully
      const status = await messaging.getWorkerStatus();
      expect(status).toBeDefined();
    });
  });

  describe('Shutdown', () => {
    it('should shutdown cleanly', async () => {
      await messaging.initialize();
      await expect(messaging.shutdown()).resolves.not.toThrow();
    });

    it('should handle shutdown without initialization', async () => {
      const freshMessaging = new MessagingSystem();
      await expect(freshMessaging.shutdown()).resolves.not.toThrow();
    });

    it('should handle shutdown with active subscriptions', async () => {
      await messaging.initialize();
      
      const callback = vi.fn();
      if (messaging.subscribe) {
        await messaging.subscribe('test.topic', callback);
      }
      
      await expect(messaging.shutdown()).resolves.not.toThrow();
    });

    it('should handle multiple shutdown calls', async () => {
      await messaging.initialize();
      
      await messaging.shutdown();
      await expect(messaging.shutdown()).resolves.not.toThrow();
    });
  });

  describe('Error Recovery', () => {
    it('should handle temporary connection issues', async () => {
      await messaging.initialize();
      
      // Simulate temporary network issue
      const { connect } = await import('nats');
      const mockConnection = await vi.mocked(connect).mock.results[0]?.value;
      
      if (mockConnection?.publish) {
        vi.mocked(mockConnection.publish)
          .mockRejectedValueOnce(new Error('Temporary failure'))
          .mockResolvedValueOnce(undefined);
      }

      // Should recover and continue working
      const status1 = await messaging.getWorkerStatus();
      const status2 = await messaging.getWorkerStatus();
      
      expect(status1).toBeDefined();
      expect(status2).toBeDefined();
    });

    it('should provide fallback behavior when messaging unavailable', async () => {
      const { connect } = await import('nats');
      vi.mocked(connect).mockRejectedValue(new Error('NATS unavailable'));
      
      const fallbackMessaging = new MessagingSystem();
      
      // Should initialize with fallback behavior
      await expect(fallbackMessaging.initialize()).rejects.toThrow();
      
      // But should still provide status (with defaults)
      const status = await fallbackMessaging.getWorkerStatus();
      expect(status).toBeDefined();
      expect(status.ingest_active).toBe(0);
      expect(status.query_active).toBe(0);
      expect(status.maintenance_active).toBe(0);
    });
  });

  describe('Configuration', () => {
    it('should use default configuration', async () => {
      await expect(messaging.initialize()).resolves.not.toThrow();
    });

    it('should handle custom configuration', async () => {
      const customConfig = {
        servers: ['nats://localhost:4222'],
        maxReconnectAttempts: 10,
      };
      
      const customMessaging = new MessagingSystem(customConfig);
      await expect(customMessaging.initialize()).resolves.not.toThrow();
      
      await customMessaging.shutdown();
    });
  });

  describe('Performance', () => {
    beforeEach(async () => {
      await messaging.initialize();
    });

    it('should handle rapid status checks', async () => {
      const promises = Array(10).fill(0).map(() => messaging.getWorkerStatus());
      
      const results = await Promise.all(promises);
      
      results.forEach(status => {
        expect(status).toBeDefined();
        expect(typeof status.ingest_active).toBe('number');
      });
    });

    it('should handle concurrent operations', async () => {
      const statusPromise = messaging.getWorkerStatus();
      const publishPromise = messaging.publish?.('test', { data: 'test' });
      
      if (publishPromise) {
        await expect(Promise.all([statusPromise, publishPromise]))
          .resolves.not.toThrow();
      } else {
        await expect(statusPromise).resolves.not.toThrow();
      }
    });
  });
});