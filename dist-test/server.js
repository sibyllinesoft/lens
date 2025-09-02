#!/usr/bin/env node
"use strict";
/**
 * Lens Search Server Entry Point
 * Initializes OpenTelemetry tracing before starting the server
 */
Object.defineProperty(exports, "__esModule", { value: true });
// Import telemetry first to initialize tracing
require("./telemetry/tracer.js");
const server_js_1 = require("./api/server.js");
// Get configuration from environment
const PORT = parseInt(process.env.PORT || '3000');
const HOST = process.env.HOST || '0.0.0.0';
// Start the server
(0, server_js_1.startServer)(PORT, HOST).catch((error) => {
    console.error('Failed to start Lens server:', error);
    process.exit(1);
});
