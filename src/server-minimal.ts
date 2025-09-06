#!/usr/bin/env node

/**
 * Minimal Lens Server for Daemon Operations
 * Basic HTTP server without complex API dependencies
 */

import { createServer, IncomingMessage, ServerResponse } from 'http';
import { URL } from 'url';

export interface ServerConfig {
  port: number;
  host: string;
}

/**
 * Basic health check endpoint
 */
function handleHealthCheck(req: IncomingMessage, res: ServerResponse): void {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify({
    status: 'ok',
    service: 'lens-daemon',
    version: '1.0.0-rc.2',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
  }));
}

/**
 * Basic status endpoint
 */
function handleStatus(req: IncomingMessage, res: ServerResponse): void {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify({
    daemon: {
      status: 'running',
      pid: process.pid,
      version: '1.0.0-rc.2',
      started: new Date().toISOString(),
      uptime: process.uptime(),
    },
    system: {
      node_version: process.version,
      platform: process.platform,
      arch: process.arch,
      memory: process.memoryUsage(),
      cpu_usage: process.cpuUsage(),
    }
  }));
}

/**
 * Handle 404 responses
 */
function handle404(req: IncomingMessage, res: ServerResponse): void {
  res.statusCode = 404;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify({
    error: 'Not Found',
    message: `The requested endpoint ${req.url} does not exist`,
    available_endpoints: ['/health', '/status'],
  }));
}

/**
 * Basic request router
 */
function handleRequest(req: IncomingMessage, res: ServerResponse): void {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  // Handle preflight OPTIONS requests
  if (req.method === 'OPTIONS') {
    res.statusCode = 200;
    res.end();
    return;
  }

  const url = new URL(req.url || '/', `http://${req.headers.host}`);
  const pathname = url.pathname;

  console.log(`${new Date().toISOString()} - ${req.method} ${pathname}`);

  // Route requests
  switch (pathname) {
    case '/health':
      handleHealthCheck(req, res);
      break;
    case '/status':
      handleStatus(req, res);
      break;
    case '/':
      // Root endpoint - redirect to status
      res.statusCode = 302;
      res.setHeader('Location', '/status');
      res.end();
      break;
    default:
      handle404(req, res);
      break;
  }
}

/**
 * Start the minimal server
 */
export async function startServer(port: number = 5678, host: string = '0.0.0.0'): Promise<void> {
  return new Promise((resolve, reject) => {
    const server = createServer(handleRequest);

    // Handle server errors
    server.on('error', (error: Error) => {
      console.error('Server error:', error);
      reject(error);
    });

    // Start listening
    server.listen(port, host, () => {
      console.log(`🚀 Lens daemon server started`);
      console.log(`📍 Listening on http://${host}:${port}`);
      console.log(`🔍 Health check: http://${host}:${port}/health`);
      console.log(`📊 Status: http://${host}:${port}/status`);
      resolve();
    });

    // Graceful shutdown handling
    process.on('SIGTERM', () => {
      console.log('📢 Received SIGTERM, shutting down gracefully...');
      server.close(() => {
        console.log('✅ Server stopped');
        process.exit(0);
      });
    });

    process.on('SIGINT', () => {
      console.log('📢 Received SIGINT, shutting down gracefully...');
      server.close(() => {
        console.log('✅ Server stopped');
        process.exit(0);
      });
    });
  });
}

// If called directly, start the server
if (require.main === module) {
  const PORT = parseInt(process.env['PORT'] || '5678');
  const HOST = process.env['HOST'] || '0.0.0.0';

  startServer(PORT, HOST).catch((error) => {
    console.error('Failed to start Lens daemon server:', error);
    process.exit(1);
  });
}