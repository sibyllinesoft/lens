#!/usr/bin/env node

/**
 * Lens Daemon Server Entry Point
 * Simplified version for daemon operations without complex API dependencies
 */

// Import minimal telemetry
import './telemetry/tracer-minimal.js';

import { startServer } from './server-minimal.js';

// Get configuration from environment
const PORT = parseInt(process.env['PORT'] || '5678');
const HOST = process.env['HOST'] || '0.0.0.0';

console.log('🚀 Starting Lens daemon server...');
console.log(`🌍 Environment: ${process.env['NODE_ENV'] || 'development'}`);
console.log(`📍 Target: ${HOST}:${PORT}`);

// Start the server
startServer(PORT, HOST).catch((error) => {
  console.error('❌ Failed to start Lens daemon server:', error);
  process.exit(1);
});