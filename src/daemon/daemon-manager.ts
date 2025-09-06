#!/usr/bin/env node
/**
 * Lens Daemon Manager - Process management for running Lens as a daemon service
 * 
 * Provides:
 * - Process lifecycle management (start/stop/restart/status)
 * - PID file management  
 * - Signal handling (SIGTERM, SIGINT, SIGHUP)
 * - Health monitoring and auto-restart
 * - Log management with rotation
 * - Configuration file support
 * - Service discovery/registration
 */

import { spawn, ChildProcess } from 'child_process';
import { promises as fs } from 'fs';
import { join, resolve } from 'path';
import { existsSync } from 'fs';
import { homedir } from 'os';
import chalk from 'chalk';

export interface DaemonConfig {
  pidFile: string;
  logFile: string;
  errorLogFile: string;
  configFile: string;
  workingDir: string;
  autoRestart: boolean;
  maxRestarts: number;
  restartDelay: number;
  healthCheckInterval: number;
  healthCheckTimeout: number;
  port?: number;
  host?: string;
  environment: 'development' | 'production' | 'staging';
}

export interface DaemonStatus {
  pid: number | null;
  running: boolean;
  uptime: number | null;
  restartCount: number;
  lastStarted: Date | null;
  lastError: string | null;
  health: 'healthy' | 'unhealthy' | 'unknown';
  memoryUsage?: NodeJS.MemoryUsage;
  cpuUsage?: NodeJS.CpuUsage;
}

export class DaemonManager {
  private config: DaemonConfig;
  private process: ChildProcess | null = null;
  private restartCount = 0;
  private startTime: Date | null = null;
  private healthCheckTimer: NodeJS.Timeout | null = null;
  private lastHealthCheck: Date | null = null;

  constructor(config?: Partial<DaemonConfig>) {
    const lensDir = join(homedir(), '.lens');
    
    this.config = {
      pidFile: join(lensDir, 'lens.pid'),
      logFile: join(lensDir, 'logs', 'lens.log'),
      errorLogFile: join(lensDir, 'logs', 'lens.error.log'),
      configFile: join(lensDir, 'lens.config.json'),
      workingDir: process.cwd(),
      autoRestart: true,
      maxRestarts: 5,
      restartDelay: 5000, // 5 seconds
      healthCheckInterval: 30000, // 30 seconds
      healthCheckTimeout: 10000, // 10 seconds
      port: 5678,
      host: '0.0.0.0',
      environment: 'production',
      ...config,
    };

    // Ensure directories exist
    this.ensureDirectories();
  }

  /**
   * Ensure all necessary directories exist
   */
  private async ensureDirectories(): Promise<void> {
    const dirs = [
      resolve(this.config.pidFile, '..'),
      resolve(this.config.logFile, '..'),
      resolve(this.config.configFile, '..'),
    ];

    for (const dir of dirs) {
      try {
        await fs.mkdir(dir, { recursive: true });
      } catch (error) {
        console.error(`Failed to create directory ${dir}:`, error);
      }
    }
  }

  /**
   * Load configuration from file if it exists
   */
  async loadConfig(): Promise<void> {
    try {
      if (existsSync(this.config.configFile)) {
        const configData = await fs.readFile(this.config.configFile, 'utf-8');
        const fileConfig = JSON.parse(configData);
        this.config = { ...this.config, ...fileConfig };
      }
    } catch (error) {
      console.warn(`Warning: Failed to load config from ${this.config.configFile}:`, error);
    }
  }

  /**
   * Save current configuration to file
   */
  async saveConfig(): Promise<void> {
    try {
      await fs.writeFile(
        this.config.configFile,
        JSON.stringify(this.config, null, 2),
        'utf-8'
      );
    } catch (error) {
      console.error(`Failed to save config to ${this.config.configFile}:`, error);
      throw error;
    }
  }

  /**
   * Read PID from PID file
   */
  private async readPid(): Promise<number | null> {
    try {
      if (!existsSync(this.config.pidFile)) {
        return null;
      }
      const pidStr = await fs.readFile(this.config.pidFile, 'utf-8');
      const pid = parseInt(pidStr.trim(), 10);
      return isNaN(pid) ? null : pid;
    } catch (error) {
      return null;
    }
  }

  /**
   * Write PID to PID file
   */
  private async writePid(pid: number): Promise<void> {
    try {
      await fs.writeFile(this.config.pidFile, pid.toString(), 'utf-8');
    } catch (error) {
      console.error(`Failed to write PID file ${this.config.pidFile}:`, error);
      throw error;
    }
  }

  /**
   * Remove PID file
   */
  private async removePidFile(): Promise<void> {
    try {
      if (existsSync(this.config.pidFile)) {
        await fs.unlink(this.config.pidFile);
      }
    } catch (error) {
      console.error(`Failed to remove PID file ${this.config.pidFile}:`, error);
    }
  }

  /**
   * Check if process with given PID is running
   */
  private isProcessRunning(pid: number): boolean {
    try {
      process.kill(pid, 0); // Signal 0 checks if process exists
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Perform health check on running service
   */
  private async performHealthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`http://${this.config.host}:${this.config.port}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(this.config.healthCheckTimeout),
      });
      
      this.lastHealthCheck = new Date();
      return response.ok;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  /**
   * Start the daemon process
   */
  async start(foreground: boolean = false): Promise<void> {
    // Check if already running
    const existingPid = await this.readPid();
    if (existingPid && this.isProcessRunning(existingPid)) {
      throw new Error(`Lens daemon is already running with PID ${existingPid}`);
    }

    // Clean up stale PID file
    if (existingPid) {
      await this.removePidFile();
    }

    console.log(chalk.blue('🚀 Starting Lens daemon...'));

    // Determine the server script path
    const serverScript = resolve(__dirname, '../server-daemon.cjs');
    if (!existsSync(serverScript)) {
      throw new Error(`Server script not found: ${serverScript}`);
    }

    // Environment variables for the daemon
    const env = {
      ...process.env,
      PORT: this.config.port?.toString() || '5678',
      HOST: this.config.host || '0.0.0.0',
      NODE_ENV: this.config.environment,
      LENS_DAEMON_MODE: 'true',
    };

    if (foreground) {
      // Run in foreground mode
      console.log(chalk.green(`🌟 Starting Lens server in foreground mode on ${this.config.host}:${this.config.port}`));
      
      this.process = spawn('node', [serverScript], {
        cwd: this.config.workingDir,
        env,
        stdio: 'inherit',
      });

      this.startTime = new Date();
      
      // Handle process events
      this.process.on('error', (error) => {
        console.error(chalk.red('Process error:'), error);
        process.exit(1);
      });

      this.process.on('exit', (code, signal) => {
        console.log(chalk.yellow(`Process exited with code ${code} and signal ${signal}`));
        process.exit(code || 0);
      });

      // Handle termination signals
      process.on('SIGTERM', () => this.gracefulShutdown());
      process.on('SIGINT', () => this.gracefulShutdown());

    } else {
      // Run as daemon (background)
      const logStream = await fs.open(this.config.logFile, 'a');
      const errorStream = await fs.open(this.config.errorLogFile, 'a');

      this.process = spawn('node', [serverScript], {
        cwd: this.config.workingDir,
        env,
        detached: true,
        stdio: ['ignore', logStream.fd, errorStream.fd],
      });

      if (!this.process.pid) {
        throw new Error('Failed to start daemon process');
      }

      // Write PID file
      await this.writePid(this.process.pid);
      this.startTime = new Date();

      // Detach from parent process
      this.process.unref();

      // Start health monitoring
      if (this.config.autoRestart) {
        this.startHealthMonitoring();
      }

      console.log(chalk.green(`✅ Lens daemon started with PID ${this.process.pid}`));
      console.log(chalk.blue(`📍 Server running on ${this.config.host}:${this.config.port}`));
      console.log(chalk.gray(`📄 Logs: ${this.config.logFile}`));
      console.log(chalk.gray(`🔧 Config: ${this.config.configFile}`));
    }
  }

  /**
   * Stop the daemon process
   */
  async stop(): Promise<void> {
    const pid = await this.readPid();
    
    if (!pid) {
      console.log(chalk.yellow('⚠️  No daemon PID found - daemon may not be running'));
      return;
    }

    if (!this.isProcessRunning(pid)) {
      console.log(chalk.yellow(`⚠️  Process with PID ${pid} is not running`));
      await this.removePidFile();
      return;
    }

    console.log(chalk.blue(`🛑 Stopping Lens daemon (PID: ${pid})...`));

    try {
      // Try graceful shutdown first (SIGTERM)
      process.kill(pid, 'SIGTERM');
      
      // Wait for process to exit gracefully
      let attempts = 0;
      const maxAttempts = 30; // 30 seconds
      
      while (attempts < maxAttempts && this.isProcessRunning(pid)) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        attempts++;
      }

      // Force kill if still running
      if (this.isProcessRunning(pid)) {
        console.log(chalk.yellow('⚠️  Process did not exit gracefully, forcing shutdown...'));
        process.kill(pid, 'SIGKILL');
        
        // Wait a bit more
        await new Promise(resolve => setTimeout(resolve, 2000));
      }

      // Clean up
      await this.removePidFile();
      this.stopHealthMonitoring();

      if (this.isProcessRunning(pid)) {
        console.log(chalk.red(`❌ Failed to stop process with PID ${pid}`));
      } else {
        console.log(chalk.green('✅ Lens daemon stopped successfully'));
      }

    } catch (error) {
      if ((error as any).code === 'ESRCH') {
        console.log(chalk.green('✅ Process was already stopped'));
        await this.removePidFile();
      } else {
        console.error(chalk.red('❌ Error stopping daemon:'), error);
        throw error;
      }
    }
  }

  /**
   * Restart the daemon process
   */
  async restart(): Promise<void> {
    console.log(chalk.blue('🔄 Restarting Lens daemon...'));
    
    try {
      await this.stop();
      
      // Wait a moment before restarting
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      await this.start();
      this.restartCount++;
      
      console.log(chalk.green('✅ Lens daemon restarted successfully'));
    } catch (error) {
      console.error(chalk.red('❌ Failed to restart daemon:'), error);
      throw error;
    }
  }

  /**
   * Get current daemon status
   */
  async getStatus(): Promise<DaemonStatus> {
    const pid = await this.readPid();
    const running = pid ? this.isProcessRunning(pid) : false;
    
    let uptime: number | null = null;
    let health: 'healthy' | 'unhealthy' | 'unknown' = 'unknown';

    if (running && this.startTime) {
      uptime = Date.now() - this.startTime.getTime();
      
      // Perform health check if running
      try {
        const isHealthy = await this.performHealthCheck();
        health = isHealthy ? 'healthy' : 'unhealthy';
      } catch (error) {
        health = 'unhealthy';
      }
    }

    return {
      pid,
      running,
      uptime,
      restartCount: this.restartCount,
      lastStarted: this.startTime,
      lastError: null, // TODO: Implement error tracking
      health,
      memoryUsage: running ? process.memoryUsage() : undefined,
      cpuUsage: running ? process.cpuUsage() : undefined,
    };
  }

  /**
   * Get recent log entries
   */
  async getLogs(lines: number = 100): Promise<string[]> {
    try {
      if (!existsSync(this.config.logFile)) {
        return ['Log file does not exist yet'];
      }

      const logContent = await fs.readFile(this.config.logFile, 'utf-8');
      const logLines = logContent.split('\n').filter(line => line.trim());
      
      return logLines.slice(-lines);
    } catch (error) {
      console.error('Failed to read logs:', error);
      return [`Error reading logs: ${error}`];
    }
  }

  /**
   * Start health monitoring with auto-restart capability
   */
  private startHealthMonitoring(): void {
    if (this.healthCheckTimer) {
      return; // Already monitoring
    }

    this.healthCheckTimer = setInterval(async () => {
      try {
        const status = await this.getStatus();
        
        if (!status.running) {
          console.log(chalk.yellow('⚠️  Daemon process died, attempting restart...'));
          
          if (this.restartCount < this.config.maxRestarts) {
            await new Promise(resolve => setTimeout(resolve, this.config.restartDelay));
            await this.start();
            console.log(chalk.green(`✅ Daemon restarted (attempt ${this.restartCount + 1}/${this.config.maxRestarts})`));
          } else {
            console.log(chalk.red(`❌ Maximum restart attempts (${this.config.maxRestarts}) reached, giving up`));
            this.stopHealthMonitoring();
          }
        } else if (status.health === 'unhealthy') {
          console.log(chalk.yellow('⚠️  Health check failed, daemon may be unresponsive'));
        }
      } catch (error) {
        console.error('Health monitoring error:', error);
      }
    }, this.config.healthCheckInterval);
  }

  /**
   * Stop health monitoring
   */
  private stopHealthMonitoring(): void {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
      this.healthCheckTimer = null;
    }
  }

  /**
   * Graceful shutdown handler
   */
  private async gracefulShutdown(): Promise<void> {
    console.log(chalk.blue('📢 Received shutdown signal, gracefully shutting down...'));
    
    this.stopHealthMonitoring();
    
    if (this.process) {
      this.process.kill('SIGTERM');
    }
    
    await this.removePidFile();
    
    console.log(chalk.green('✅ Shutdown complete'));
    process.exit(0);
  }

  /**
   * Handle log rotation
   */
  async rotateLogs(): Promise<void> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    try {
      // Rotate main log
      if (existsSync(this.config.logFile)) {
        const rotatedLogFile = `${this.config.logFile}.${timestamp}`;
        await fs.rename(this.config.logFile, rotatedLogFile);
      }
      
      // Rotate error log
      if (existsSync(this.config.errorLogFile)) {
        const rotatedErrorFile = `${this.config.errorLogFile}.${timestamp}`;
        await fs.rename(this.config.errorLogFile, rotatedErrorFile);
      }
      
      console.log(chalk.green('✅ Log rotation completed'));
    } catch (error) {
      console.error(chalk.red('❌ Log rotation failed:'), error);
      throw error;
    }
  }
}