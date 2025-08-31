/**
 * Fastify server for Lens API
 * Implements all endpoints with OpenTelemetry tracing and validation
 */

import Fastify from 'fastify';
import cors from '@fastify/cors';
import { v4 as uuidv4 } from 'uuid';
import {
  SearchRequestSchema,
  SearchResponseSchema,
  StructRequestSchema,
  SymbolsNearRequestSchema,
  HealthResponseSchema,
  type SearchRequest,
  type SearchResponse,
  type StructRequest,
  type SymbolsNearRequest,
  type HealthResponse,
} from '../types/api.js';
import { LensTracer } from '../telemetry/tracer.js';
import { LensSearchEngine } from './search-engine.js';
import { PRODUCTION_CONFIG } from '../types/config.js';
import { registerBenchmarkEndpoints } from './benchmark-endpoints.js';

const fastify = Fastify({
  logger: {
    level: 'info',
    serializers: {
      req(request) {
        const result: any = {
          method: request.method,
          url: request.url,
          headers: request.headers,
          hostname: request.hostname,
          remoteAddress: request.ip,
        };
        if (request.socket?.remotePort) {
          result.remotePort = request.socket.remotePort;
        }
        return result;
      },
    },
  },
});

// CORS support
fastify.register(cors, {
  origin: true,
});

// Global error handler
fastify.setErrorHandler((error, request, reply) => {
  fastify.log.error(error);
  
  // Extract trace ID from headers if available
  const traceId = request.headers['x-trace-id'] as string || uuidv4();
  
  // Handle different error types appropriately
  let statusCode = 500;
  let errorMessage = error.message;
  
  // Fastify-specific errors
  if ('statusCode' in error && typeof error.statusCode === 'number') {
    statusCode = error.statusCode;
  }
  
  // Malformed JSON (syntax error)
  if (error instanceof SyntaxError && error.message.includes('JSON')) {
    statusCode = 400;
    errorMessage = 'Invalid JSON syntax';
  }
  
  // Content-Type validation errors (Fastify errors)
  if (error.message.includes('Unsupported Media Type') || 
      (error as any).code === 'FST_ERR_CTP_INVALID_MEDIA_TYPE') {
    statusCode = 415;
    errorMessage = 'Unsupported Media Type';
  }
  
  // Validation errors (from Zod schemas)
  if (error.message.includes('validation') || error.name === 'ZodError') {
    statusCode = 400;
    errorMessage = 'Invalid request format';
  }
  
  reply.status(statusCode).send({
    error: statusCode === 400 ? 'Bad Request' : statusCode === 415 ? 'Unsupported Media Type' : 'Internal Server Error',
    message: errorMessage,
    trace_id: traceId,
    timestamp: new Date().toISOString(),
  });
});

// Initialize search engine
const searchEngine = new LensSearchEngine();

// Initialize the search engine (async initialization)
async function initializeServer() {
  try {
    await searchEngine.initialize();
    console.log('🔍 Lens Search Engine initialized for API server');
  } catch (error) {
    console.error('Failed to initialize search engine:', error);
    throw error;
  }
}

// Track initialization state
export let isInitialized = false;

// Update initialization function to set the flag
const originalInitialize = initializeServer;
async function initializeServerWithTracking() {
  if (!isInitialized) {
    await originalInitialize();
    // @ts-ignore - need to mutate exported variable
    isInitialized = true;
  }
}

// Override the export
export { initializeServerWithTracking as initializeServer };

// Health check endpoint
fastify.get('/health', async (request, reply): Promise<HealthResponse> => {
  const span = LensTracer.createChildSpan('health_check');
  const startTime = Date.now();

  try {
    const health = await searchEngine.getHealthStatus();
    const latency = Date.now() - startTime;
    
    // Check SLA compliance (5ms target)
    if (latency > PRODUCTION_CONFIG.api_limits.rate_limit_per_sec) {
      fastify.log.warn(`Health check SLA breach: ${latency}ms > 5ms`);
    }

    span.setAttributes({
      success: true,
      latency_ms: latency,
      shards_healthy: health.shards_healthy,
      status: health.status,
    });

    const response: HealthResponse = {
      status: health.status,
      timestamp: new Date().toISOString(),
      shards_healthy: health.shards_healthy,
    };

    // Validate response against schema
    HealthResponseSchema.parse(response);
    
    return response;

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    const response: HealthResponse = {
      status: 'down',
      timestamp: new Date().toISOString(),
      shards_healthy: 0,
    };
    
    reply.status(503);
    return response;
    
  } finally {
    span.end();
  }
});

// Main search endpoint
fastify.post('/search', async (request, reply): Promise<SearchResponse> => {
  const span = LensTracer.createChildSpan('api_search');
  const startTime = Date.now();

  try {
    // Validate request
    const searchRequest = SearchRequestSchema.parse(request.body);
    const traceId = request.headers['x-trace-id'] as string || uuidv4();

    span.setAttributes({
      'request.query': searchRequest.q,
      'request.mode': searchRequest.mode,
      'request.fuzzy': searchRequest.fuzzy,
      'request.k': searchRequest.k,
      'trace_id': traceId,
    });

    // Perform search
    const result = await searchEngine.search({
      trace_id: traceId,
      query: searchRequest.q,
      mode: searchRequest.mode,
      k: searchRequest.k,
      fuzzy_distance: searchRequest.fuzzy,
      started_at: new Date(),
      stages: [],
    });

    const totalLatency = Date.now() - startTime;

    // Check SLA compliance (20ms target)
    if (totalLatency > 20) {
      fastify.log.warn(`Search SLA breach: ${totalLatency}ms > 20ms`);
    }

    // Build response
    const response: SearchResponse = {
      hits: result.candidates.map(candidate => ({
        file: candidate.file_path,
        line: candidate.line,
        col: candidate.col,
        ast_path: candidate.ast_path,
        symbol_kind: candidate.symbol_kind,
        score: candidate.score,
        why: candidate.match_reasons,
      })),
      total: result.candidates.length,
      latency_ms: {
        stage_a: result.stage_a_latency || 0,
        stage_b: result.stage_b_latency || 0,
        stage_c: result.stage_c_latency,
        total: totalLatency,
      },
      trace_id: traceId,
    };

    // Validate response
    SearchResponseSchema.parse(response);

    span.setAttributes({
      success: true,
      hits_count: response.hits.length,
      total_latency_ms: totalLatency,
      stage_a_latency_ms: response.latency_ms.stage_a,
      stage_b_latency_ms: response.latency_ms.stage_b,
      stage_c_latency_ms: response.latency_ms.stage_c || 0,
    });

    return response;

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    const traceId = request.headers['x-trace-id'] as string || uuidv4();
    
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    fastify.log.error(`Search failed: ${errorMsg} (trace: ${traceId})`);

    // Return error response
    reply.status(400);
    return {
      hits: [],
      total: 0,
      latency_ms: {
        stage_a: 0,
        stage_b: 0,
        total: Date.now() - startTime,
      },
      trace_id: traceId,
    };

  } finally {
    span.end();
  }
});

// Structural search endpoint
fastify.post('/struct', async (request, reply): Promise<SearchResponse> => {
  const span = LensTracer.createChildSpan('api_struct');
  const startTime = Date.now();

  try {
    // Validate request
    const structRequest = StructRequestSchema.parse(request.body);
    const traceId = request.headers['x-trace-id'] as string || uuidv4();

    span.setAttributes({
      'request.pattern': structRequest.pattern,
      'request.lang': structRequest.lang,
      'request.max_results': structRequest.max_results || 100,
      'trace_id': traceId,
    });

    // Perform structural search
    const result = await searchEngine.structuralSearch({
      trace_id: traceId,
      query: structRequest.pattern,
      mode: 'struct',
      k: structRequest.max_results || 100,
      fuzzy_distance: 0,
      started_at: new Date(),
      stages: [],
    }, structRequest.lang);

    const totalLatency = Date.now() - startTime;

    // Check SLA compliance (30ms target)
    if (totalLatency > 30) {
      fastify.log.warn(`Struct search SLA breach: ${totalLatency}ms > 30ms`);
    }

    const response: SearchResponse = {
      hits: result.candidates.map(candidate => ({
        file: candidate.file_path,
        line: candidate.line,
        col: candidate.col,
        ast_path: candidate.ast_path,
        symbol_kind: candidate.symbol_kind,
        score: candidate.score,
        why: candidate.match_reasons,
      })),
      total: result.candidates.length,
      latency_ms: {
        stage_a: result.stage_a_latency || 0,
        stage_b: result.stage_b_latency || 0,
        stage_c: result.stage_c_latency,
        total: totalLatency,
      },
      trace_id: traceId,
    };

    SearchResponseSchema.parse(response);
    
    span.setAttributes({
      success: true,
      hits_count: response.hits.length,
      total_latency_ms: totalLatency,
    });

    return response;

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    const traceId = request.headers['x-trace-id'] as string || uuidv4();
    
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(400);
    return {
      hits: [],
      total: 0,
      latency_ms: {
        stage_a: 0,
        stage_b: 0,
        total: Date.now() - startTime,
      },
      trace_id: traceId,
    };

  } finally {
    span.end();
  }
});

// Symbols near endpoint
fastify.post('/symbols/near', async (request, reply): Promise<SearchResponse> => {
  const span = LensTracer.createChildSpan('api_symbols_near');
  const startTime = Date.now();

  try {
    // Validate request
    const symbolsRequest = SymbolsNearRequestSchema.parse(request.body);
    const traceId = request.headers['x-trace-id'] as string || uuidv4();

    span.setAttributes({
      'request.file': symbolsRequest.file,
      'request.line': symbolsRequest.line,
      'request.radius': symbolsRequest.radius || 25,
      'trace_id': traceId,
    });

    // Find symbols near location
    const result = await searchEngine.findSymbolsNear(
      symbolsRequest.file,
      symbolsRequest.line,
      symbolsRequest.radius || 25
    );

    const totalLatency = Date.now() - startTime;

    // Check SLA compliance (15ms target)
    if (totalLatency > 15) {
      fastify.log.warn(`Symbols near SLA breach: ${totalLatency}ms > 15ms`);
    }

    const response: SearchResponse = {
      hits: result.map(candidate => ({
        file: candidate.file_path,
        line: candidate.line,
        col: candidate.col,
        ast_path: candidate.ast_path,
        symbol_kind: candidate.symbol_kind,
        score: candidate.score,
        why: candidate.match_reasons,
      })),
      total: result.length,
      latency_ms: {
        stage_a: 0,
        stage_b: totalLatency, // Symbols are Stage B
        total: totalLatency,
      },
      trace_id: traceId,
    };

    SearchResponseSchema.parse(response);
    
    span.setAttributes({
      success: true,
      hits_count: response.hits.length,
      total_latency_ms: totalLatency,
    });

    return response;

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    const traceId = request.headers['x-trace-id'] as string || uuidv4();
    
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(400);
    return {
      hits: [],
      total: 0,
      latency_ms: {
        stage_a: 0,
        stage_b: Date.now() - startTime,
        total: Date.now() - startTime,
      },
      trace_id: traceId,
    };

  } finally {
    span.end();
  }
});

// Start server
export async function startServer(port: number = 3000, host: string = '0.0.0.0') {
  try {
    // Initialize search engine
    await searchEngine.initialize();
    
    // Register benchmark endpoints
    await registerBenchmarkEndpoints(fastify);
    
    await fastify.listen({ port, host });
    console.log(`🚀 Lens server running on http://${host}:${port}`);
    console.log(`📊 Metrics available at http://${host}:9464/metrics`);
    console.log(`🔍 Health check at http://${host}:${port}/health`);
    console.log(`🧪 Benchmark endpoints at http://${host}:${port}/bench/`);
    
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
}

// Graceful shutdown
['SIGINT', 'SIGTERM'].forEach((signal) => {
  process.on(signal, async () => {
    console.log(`\nReceived ${signal}, shutting down gracefully...`);
    
    try {
      await searchEngine.shutdown();
      await fastify.close();
      console.log('Server shut down successfully');
      process.exit(0);
    } catch (err) {
      console.error('Error during shutdown:', err);
      process.exit(1);
    }
  });
});

export { fastify, searchEngine };