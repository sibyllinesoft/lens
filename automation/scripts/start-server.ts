#!/usr/bin/env node

/**
 * Simple server starter that calls the startServer function
 */

// Set environment variables before importing to avoid LSP initialization
process.env.LENS_DISABLE_LSP = 'true';

import { startServer } from './src/api/server.js';

async function main() {
  try {
    console.log('🔥 Starting Lens search server...');
    await startServer(3000, '0.0.0.0');
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
}

main();