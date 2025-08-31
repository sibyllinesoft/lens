#!/usr/bin/env node

/**
 * Lens Search Server Entry Point
 * Initializes OpenTelemetry tracing before starting the server
 */

// Import telemetry first to initialize tracing
import './telemetry/tracer.js';

import { startServer } from './api/server.js';

// Get configuration from environment
const PORT = parseInt(process.env['PORT'] || '3000');
const HOST = process.env['HOST'] || '0.0.0.0';

// Start the server
startServer(PORT, HOST).catch((error) => {
  console.error('Failed to start Lens server:', error);
  process.exit(1);
});