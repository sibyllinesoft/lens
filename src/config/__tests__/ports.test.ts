/**
 * Comprehensive tests for Port Manager
 * Covers dynamic port allocation, configuration persistence, and environment variables
 */

import { describe, it, expect, beforeEach, afterEach, mock, jest, mock } from 'bun:test';
import * as fs from 'fs/promises';
import * as net from 'net';
import * as path from 'path';
import { portManager, getServerPort, getMetricsPort, getApiUrl, type PortConfig } from '../ports.js';

// Mock fs operations
mock('fs/promises');
mock('net');

const mockFs = mocked(fs);
const mockNet = mocked(net);

// Mock server for port testing
const createMockServer = (shouldError = false) => {
  const server = {
    listen: jest.fn().mockImplementation((port, callback) => {
      if (shouldError) {
        setImmediate(() => server.emit('error', new Error('Port in use')));
      } else {
        setImmediate(callback);
      }
    }),
    close: jest.fn().mockImplementation((callback) => {
      setImmediate(callback, true);
    }),
    on: jest.fn().mockImplementation((event, handler) => {
      if (event === 'error' && shouldError) {
        setImmediate(handler);
      }
    }),
    emit: jest.fn(),
  };
  return server;
};

describe('Port Manager', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset the singleton's internal state
    (portManager as any).config = null;
    
    // Mock process.cwd()
    jest.spyOn(process, 'cwd').mockReturnValue('/test');
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Port Availability Checking', () => {
    it('should detect available port', async () => {
      const mockServer = createMockServer(false);
      mockNet.createServer.mockReturnValue(mockServer as any);

      const config = await portManager.loadConfig();
      
      expect(mockNet.createServer).toHaveBeenCalled();
      expect(mockServer.listen).toHaveBeenCalled();
      expect(config).toHaveProperty('api_server');
      expect(config).toHaveProperty('metrics_server');
    });

    it('should handle port unavailable', async () => {
      // First port fails, second succeeds
      let callCount = 0;
      mockNet.createServer.mockImplementation(() => {
        callCount++;
        return createMockServer(callCount === 1);
      });

      const config = await portManager.loadConfig();
      
      expect(config.api_server).toBeGreaterThan(3000); // Should skip first port
    });

    it('should throw error when no ports available', async () => {
      // All ports fail
      mockNet.createServer.mockImplementation(() => createMockServer(true));

      await expect(portManager.loadConfig()).rejects.toThrow('No available ports in range');
    });
  });

  describe('Configuration Loading', () => {
    it('should load existing valid configuration', async () => {
      const existingConfig: PortConfig = {
        api_server: 3001,
        metrics_server: 3002,
        reserved_until: Date.now() + 30000, // 30 seconds from now
      };

      mockFs.readFile.mockResolvedValue(JSON.stringify(existingConfig));
      
      // Mock ports as available
      mockNet.createServer.mockImplementation(() => createMockServer(false));

      const config = await portManager.loadConfig();
      
      expect(config).toEqual(existingConfig);
      expect(mockFs.readFile).toHaveBeenCalledWith(
        expect.stringContaining('.port-config.json'),
        'utf-8'
      );
    });

    it('should reallocate when reservation expired', async () => {
      const expiredConfig: PortConfig = {
        api_server: 3001,
        metrics_server: 3002,
        reserved_until: Date.now() - 10000, // Expired 10 seconds ago
      };

      mockFs.readFile.mockResolvedValue(JSON.stringify(expiredConfig));
      mockFs.writeFile.mockResolvedValue(undefined);
      mockNet.createServer.mockImplementation(() => createMockServer(false));

      const consoleSpy = jest.spyOn(console, 'log').mockImplementation(() => {});

      const config = await portManager.loadConfig();
      
      expect(consoleSpy).toHaveBeenCalledWith('🔄 Port reservation expired, reallocating...');
      expect(config.reserved_until).toBeGreaterThan(Date.now());
      
      consoleSpy.mockRestore();
    });

    it('should reallocate when reserved ports unavailable', async () => {
      const existingConfig: PortConfig = {
        api_server: 3001,
        metrics_server: 3002,
        reserved_until: Date.now() + 30000,
      };

      mockFs.readFile.mockResolvedValue(JSON.stringify(existingConfig));
      mockFs.writeFile.mockResolvedValue(undefined);
      
      // Mock first two calls (checking existing ports) as unavailable
      // Then mock subsequent calls as available for new allocation
      let callCount = 0;
      mockNet.createServer.mockImplementation(() => {
        callCount++;
        return createMockServer(callCount <= 2); // First 2 calls fail
      });

      const consoleSpy = jest.spyOn(console, 'log').mockImplementation(() => {});

      const config = await portManager.loadConfig();
      
      expect(consoleSpy).toHaveBeenCalledWith('🔄 Reserved ports no longer available, reallocating...');
      
      consoleSpy.mockRestore();
    });

    it('should create new config when file does not exist', async () => {
      mockFs.readFile.mockRejectedValue(new Error('ENOENT: no such file'));
      mockFs.writeFile.mockResolvedValue(undefined);
      mockNet.createServer.mockImplementation(() => createMockServer(false));

      const consoleSpy = jest.spyOn(console, 'log').mockImplementation(() => {});

      const config = await portManager.loadConfig();
      
      expect(config).toHaveProperty('api_server');
      expect(config).toHaveProperty('metrics_server');
      expect(config.api_server).toBeGreaterThanOrEqual(3000);
      expect(config.metrics_server).toBeGreaterThan(config.api_server);
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringMatching(/🚀 Allocated ports - API: \d+, Metrics: \d+/)
      );
      
      consoleSpy.mockRestore();
    });

    it('should handle malformed JSON config', async () => {
      mockFs.readFile.mockResolvedValue('invalid json {');
      mockFs.writeFile.mockResolvedValue(undefined);
      mockNet.createServer.mockImplementation(() => createMockServer(false));

      const config = await portManager.loadConfig();
      
      expect(config).toHaveProperty('api_server');
      expect(mockFs.writeFile).toHaveBeenCalled();
    });
  });

  describe('Configuration Persistence', () => {
    it('should save configuration to file', async () => {
      mockFs.writeFile.mockResolvedValue(undefined);
      mockNet.createServer.mockImplementation(() => createMockServer(false));

      await portManager.loadConfig();
      
      expect(mockFs.writeFile).toHaveBeenCalledWith(
        expect.stringContaining('.port-config.json'),
        expect.stringContaining('api_server'),
        
      );
    });

    it('should extend reservation', async () => {
      // First load a config
      mockFs.readFile.mockRejectedValue(new Error('ENOENT'));
      mockFs.writeFile.mockResolvedValue(undefined);
      mockNet.createServer.mockImplementation(() => createMockServer(false));

      await portManager.loadConfig();
      
      // Clear the mock calls from initial load
      mockFs.writeFile.mockClear();

      // Now extend reservation
      await portManager.extendReservation();
      
      expect(mockFs.writeFile).toHaveBeenCalled();
      const writeCall = mockFs.writeFile.mock.calls[0];
      const configData = JSON.parse(writeCall[1] as string);
      expect(configData.reserved_until).toBeGreaterThan(Date.now());
    });

    it('should handle extend reservation when no config loaded', async () => {
      await portManager.extendReservation();
      
      // Should not throw error or call writeFile
      expect(mockFs.writeFile).not.toHaveBeenCalled();
    });
  });

  describe('Port Release', () => {
    it('should release reservations and delete config file', async () => {
      mockFs.unlink.mockResolvedValue(undefined);
      
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation(() => {});

      await portManager.releaseReservations();
      
      expect(mockFs.unlink).toHaveBeenCalledWith(expect.stringContaining('.port-config.json'));
      expect(consoleSpy).toHaveBeenCalledWith('🔓 Released port reservations');
      
      consoleSpy.mockRestore();
    });

    it('should handle release when file does not exist', async () => {
      mockFs.unlink.mockRejectedValue(new Error('ENOENT: no such file'));
      
      // Should not throw
      await expect(portManager.releaseReservations()).resolves.not.toThrow();
    });
  });

  describe('API URL Generation', () => {
    it('should generate correct API URL', async () => {
      const mockConfig: PortConfig = {
        api_server: 3001,
        metrics_server: 3002,
      };

      mockFs.readFile.mockResolvedValue(JSON.stringify(mockConfig));
      mockNet.createServer.mockImplementation(() => createMockServer(false));

      const url = await getApiUrl();
      
      expect(url).toBe('http://localhost:3001');
    });
  });

  describe('Environment Variable Integration', () => {
    it('should use PORT environment variable for server port', async () => {
      process.env.PORT = '8080';

      const port = await getServerPort();
      
      expect(port).toBe(8080);
      
      delete process.env.PORT;
    });

    it('should use METRICS_PORT environment variable for metrics port', async () => {
      process.env.METRICS_PORT = '9090';

      const port = await getMetricsPort();
      
      expect(port).toBe(9090);
      
      delete process.env.METRICS_PORT;
    });

    it('should fall back to config when environment variables not set', async () => {
      const mockConfig: PortConfig = {
        api_server: 3001,
        metrics_server: 3002,
      };

      mockFs.readFile.mockResolvedValue(JSON.stringify(mockConfig));
      mockNet.createServer.mockImplementation(() => createMockServer(false));

      const serverPort = await getServerPort();
      const metricsPort = await getMetricsPort();
      
      expect(serverPort).toBe(3001);
      expect(metricsPort).toBe(3002);
    });

    it('should handle invalid PORT environment variable', async () => {
      process.env.PORT = 'not-a-number';

      const port = await getServerPort();
      
      expect(Number.isNaN(port)).toBe(true);
      
      delete process.env.PORT;
    });
  });

  describe('Configuration Caching', () => {
    it('should return cached config on subsequent calls', async () => {
      const mockConfig: PortConfig = {
        api_server: 3001,
        metrics_server: 3002,
        reserved_until: Date.now() + 30000,
      };

      mockFs.readFile.mockResolvedValue(JSON.stringify(mockConfig));
      mockNet.createServer.mockImplementation(() => createMockServer(false));

      // First call should read file
      const config1 = await portManager.getConfig();
      
      // Second call should use cache
      const config2 = await portManager.getConfig();
      
      expect(config1).toEqual(config2);
      expect(mockFs.readFile).toHaveBeenCalledTimes(1); // Only called once
    });
  });

  describe('Edge Cases', () => {
    it('should handle rapid successive calls', async () => {
      mockFs.readFile.mockRejectedValue(new Error('ENOENT'));
      mockFs.writeFile.mockResolvedValue(undefined);
      mockNet.createServer.mockImplementation(() => createMockServer(false));

      // Make multiple simultaneous calls
      const promises = Array(5).fill(0).map(() => portManager.loadConfig());
      const results = await Promise.all(promises);
      
      // All should return valid configurations
      results.forEach(config => {
        expect(config).toHaveProperty('api_server');
        expect(config).toHaveProperty('metrics_server');
      });
    });

    it('should handle port allocation when many ports are taken', async () => {
      // Mock first 100 ports as unavailable, then available
      let callCount = 0;
      mockNet.createServer.mockImplementation(() => {
        callCount++;
        return createMockServer(callCount <= 100);
      });

      mockFs.writeFile.mockResolvedValue(undefined);

      const config = await portManager.loadConfig();
      
      expect(config.api_server).toBeGreaterThanOrEqual(3100); // Should skip first 100
    });
  });
});