/**
 * Dynamic port allocation and configuration synchronization
 * Prevents port conflicts and provides centralized port management
 */

import * as net from 'net';
import * as fs from 'fs/promises';
import * as path from 'path';

export interface PortConfig {
  api_server: number;
  metrics_server: number;
  benchmark_port?: number;
  reserved_until?: number; // timestamp when reservation expires
}

const PORT_CONFIG_FILE = path.join(process.cwd(), '.port-config.json');
const DEFAULT_PORT_RANGE = { min: 3000, max: 9000 };
const RESERVATION_DURATION = 60000; // 1 minute

class PortManager {
  private config: PortConfig | null = null;

  /**
   * Check if a port is available for use
   */
  private async isPortAvailable(port: number): Promise<boolean> {
    return new Promise((resolve) => {
      const server = net.createServer();
      server.listen(port, () => {
        server.close(() => resolve(true));
      });
      server.on('error', () => resolve(false));
    });
  }

  /**
   * Find next available port starting from a base port
   */
  private async findAvailablePort(startPort: number = DEFAULT_PORT_RANGE.min): Promise<number> {
    for (let port = startPort; port <= DEFAULT_PORT_RANGE.max; port++) {
      if (await this.isPortAvailable(port)) {
        return port;
      }
    }
    throw new Error(`No available ports in range ${DEFAULT_PORT_RANGE.min}-${DEFAULT_PORT_RANGE.max}`);
  }

  /**
   * Load existing port configuration or create new one
   */
  async loadConfig(): Promise<PortConfig> {
    try {
      const data = await fs.readFile(PORT_CONFIG_FILE, 'utf-8');
      const config: PortConfig = JSON.parse(data);
      
      // Check if reservation has expired
      if (config.reserved_until && Date.now() > config.reserved_until) {
        console.log('🔄 Port reservation expired, reallocating...');
        return this.allocateNewConfig();
      }

      // Verify ports are still available
      const apiAvailable = await this.isPortAvailable(config.api_server);
      const metricsAvailable = await this.isPortAvailable(config.metrics_server);
      
      if (!apiAvailable || !metricsAvailable) {
        console.log('🔄 Reserved ports no longer available, reallocating...');
        return this.allocateNewConfig();
      }

      this.config = config;
      return config;
      
    } catch (error) {
      // File doesn't exist or is invalid, create new config
      return this.allocateNewConfig();
    }
  }

  /**
   * Allocate new port configuration
   */
  private async allocateNewConfig(): Promise<PortConfig> {
    const apiPort = await this.findAvailablePort(3000);
    const metricsPort = await this.findAvailablePort(apiPort + 1);
    
    const config: PortConfig = {
      api_server: apiPort,
      metrics_server: metricsPort,
      reserved_until: Date.now() + RESERVATION_DURATION,
    };

    await this.saveConfig(config);
    this.config = config;
    
    console.log(`🚀 Allocated ports - API: ${apiPort}, Metrics: ${metricsPort}`);
    return config;
  }

  /**
   * Save configuration to file
   */
  private async saveConfig(config: PortConfig): Promise<void> {
    await fs.writeFile(PORT_CONFIG_FILE, JSON.stringify(config, null, 2));
  }

  /**
   * Get current port configuration
   */
  async getConfig(): Promise<PortConfig> {
    if (!this.config) {
      return this.loadConfig();
    }
    return this.config;
  }

  /**
   * Extend reservation for current ports
   */
  async extendReservation(): Promise<void> {
    if (this.config) {
      this.config.reserved_until = Date.now() + RESERVATION_DURATION;
      await this.saveConfig(this.config);
    }
  }

  /**
   * Release port reservations
   */
  async releaseReservations(): Promise<void> {
    try {
      await fs.unlink(PORT_CONFIG_FILE);
      console.log('🔓 Released port reservations');
    } catch (error) {
      // File might not exist, ignore error
    }
    this.config = null;
  }

  /**
   * Get the API server URL for the current configuration
   */
  async getApiUrl(): Promise<string> {
    const config = await this.getConfig();
    return `http://localhost:${config.api_server}`;
  }
}

// Singleton instance
export const portManager = new PortManager();

/**
 * Environment variable compatibility
 */
export async function getServerPort(): Promise<number> {
  if (process.env.PORT) {
    return parseInt(process.env.PORT, 10);
  }
  const config = await portManager.getConfig();
  return config.api_server;
}

export async function getMetricsPort(): Promise<number> {
  if (process.env.METRICS_PORT) {
    return parseInt(process.env.METRICS_PORT, 10);
  }
  const config = await portManager.getConfig();
  return config.metrics_server;
}

export async function getApiUrl(): Promise<string> {
  return portManager.getApiUrl();
}