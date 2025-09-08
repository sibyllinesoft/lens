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
  CompatibilityCheckRequestSchema,
  CompatibilityCheckResponseSchema,
  SpiSearchRequestSchema,
  SpiSearchResponseSchema,
  SpiHealthResponseSchema,
  ResolveRequestSchema,
  ResolveResponseSchema,
  ContextRequestSchema,
  ContextResponseSchema,
  XrefRequestSchema,
  XrefResponseSchema,
  SymbolsListRequestSchema,
  SymbolsListResponseSchema,
  LSPRenameRequestSchema,
  LSPRenameResponseSchema,
  LSPCodeActionsRequestSchema,
  LSPCodeActionsResponseSchema,
  LSPHierarchyRequestSchema,
  LSPHierarchyResponseSchema,
  type SearchRequest,
  type SearchResponse,
  type StructRequest,
  type SymbolsNearRequest,
  type HealthResponse,
  type CompatibilityCheckRequest,
  type CompatibilityCheckResponse,
  type SpiSearchRequest,
  type SpiSearchResponse,
  type SpiHealthResponse,
  type ResolveRequest,
  type ResolveResponse,
  type ContextRequest,
  type ContextResponse,
  type XrefRequest,
  type XrefResponse,
  type SymbolsListRequest,
  type SymbolsListResponse,
  type LSPRenameResponse,
  type LSPCodeActionsResponse,
  type LSPHierarchyResponse,
} from '../types/api.js';
import { LensTracer } from '../telemetry/tracer.js';
import { LensSearchEngine } from './search-engine.js';
import { PRODUCTION_CONFIG } from '../types/config.js';
import { registerBenchmarkEndpoints } from './benchmark-endpoints.js';
import { MetricsTelemetry } from '../raptor/metrics-telemetry.js';
import { registerRaptorMetricsEndpoints } from '../raptor/metrics-endpoints.js';
import { ConfigRolloutManager } from '../raptor/config-rollout.js';
import { registerConfigEndpoints } from '../raptor/config-endpoints.js';
import { registerPrecisionMonitoringEndpoints } from './precision-monitoring-endpoints.js';
import { checkCompatibility, getVersionInfo } from '../core/version-manager.js';
import { checkBundleCompatibility } from '../core/compatibility-checker.js';
import { globalFeatureFlags } from '../core/feature-flags.js';
import { runQualityGates } from '../core/quality-gates.js';
import { runNightlyValidation, getValidationStatus, globalThreeNightValidation } from '../core/three-night-validation.js';
import { getDashboardState } from '../monitoring/phase-d-dashboards.js';
import { globalAdaptiveFanout } from '../core/adaptive-fanout.js';
import { globalExperimentFramework } from '../core/precision-optimization.js';

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

// Initialize search engine with LSP support (enabled by default)
const enableLSP = process.env.LENS_DISABLE_LSP !== 'true';
const searchEngine = new LensSearchEngine(
  './indexed-content',
  undefined, // rerank config
  undefined, // phase B config
  enableLSP
);

// Initialize the search engine (async initialization)
async function initializeServer() {
  try {
    await searchEngine.initialize();
    console.log(`🔍 Lens Search Engine initialized for API server (LSP: ${enableLSP ? 'enabled' : 'disabled'})`);
    return fastify; // Return the Fastify app instance
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
    const app = await originalInitialize();
    // @ts-ignore - need to mutate exported variable
    isInitialized = true;
    return app;
  }
  return fastify;
}

// Override the export
export { initializeServerWithTracking as initializeServer };

// Manifest endpoint - maps repo_ref to repo_sha
fastify.get('/manifest', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_manifest');
  const startTime = Date.now();

  try {
    const manifest = await searchEngine.getManifest();
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      repos_count: Object.keys(manifest).length,
    });

    return manifest;

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(503);
    return { error: 'Failed to get manifest' };
    
  } finally {
    span.end();
  }
});

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

// Compatibility check endpoint
fastify.get('/compat/check', async (request, reply): Promise<CompatibilityCheckResponse> => {
  const span = LensTracer.createChildSpan('compat_check');
  const startTime = Date.now();

  try {
    // Parse query parameters
    const { api_version, index_version, policy_version, allow_compat } = request.query as any;
    
    const compatRequest: CompatibilityCheckRequest = CompatibilityCheckRequestSchema.parse({
      api_version: api_version || 'v1',
      index_version: index_version || 'v1',
      policy_version: policy_version || undefined,
      allow_compat: allow_compat === 'true' || allow_compat === true,
    });

    const result = checkCompatibility(
      compatRequest.api_version,
      compatRequest.index_version,
      compatRequest.allow_compat,
      compatRequest.policy_version
    );

    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      compatible: result.compatible,
      client_api_version: compatRequest.api_version,
      client_index_version: compatRequest.index_version,
      allow_compat: compatRequest.allow_compat,
    });

    // Check SLA compliance (10ms target)
    if (latency > 10) {
      fastify.log.warn(`Compat check SLA breach: ${latency}ms > 10ms`);
    }

    // Validate response against schema
    CompatibilityCheckResponseSchema.parse(result);
    
    return result;

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(400);
    throw error;
    
  } finally {
    span.end();
  }
});

// Bundle compatibility check endpoint
fastify.get('/compat/bundles', async (request, reply) => {
  const span = LensTracer.createChildSpan('compat_check_bundles');
  const startTime = Date.now();

  try {
    // Parse query parameters
    const { allow_compat, bundles_path } = request.query as any;
    
    const allowCompatFlag = allow_compat === 'true' || allow_compat === true;
    const bundlesPath = bundles_path || './nightly-bundles';
    
    const report = await checkBundleCompatibility(bundlesPath, allowCompatFlag);
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      bundles_checked: report.bundles_checked.length,
      compatible: report.compatible,
      overall_status: report.overall_status,
      allow_compat: allowCompatFlag,
    });

    // Check SLA compliance (50ms target for bundle checks)
    if (latency > 50) {
      fastify.log.warn(`Bundle compat check SLA breach: ${latency}ms > 50ms`);
    }

    return report;

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      compatible: false,
      error: 'Bundle compatibility check failed',
      message: errorMsg,
    };
    
  } finally {
    span.end();
  }
});

// Phase D Canary Deployment Endpoints
fastify.get('/canary/status', async (request, reply) => {
  const span = LensTracer.createChildSpan('canary_status');
  const startTime = Date.now();

  try {
    const status = globalFeatureFlags.getCanaryStatus();
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      traffic_percentage: status.trafficPercentage,
      kill_switch_enabled: status.killSwitchEnabled,
      stage_a_enabled: status.stageFlags.stageA_native_scanner,
      stage_b_enabled: status.stageFlags.stageB_enabled,
      stage_c_enabled: status.stageFlags.stageC_enabled,
    });

    return {
      success: true,
      canary_deployment: status,
      timestamp: new Date().toISOString(),
      latency_ms: latency
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to get canary status',
      message: errorMsg
    };
    
  } finally {
    span.end();
  }
});

fastify.post('/canary/progress', async (request, reply) => {
  const span = LensTracer.createChildSpan('canary_progress');
  const startTime = Date.now();

  try {
    const result = globalFeatureFlags.progressCanaryRollout();
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: result.success,
      latency_ms: latency,
      new_percentage: result.newPercentage,
      stage: result.stage,
    });

    if (!result.success) {
      reply.status(400);
      return {
        success: false,
        error: 'Cannot progress rollout',
        current_percentage: result.newPercentage,
        stage: result.stage
      };
    }

    return {
      success: true,
      message: `Canary rollout progressed to ${result.newPercentage}% (${result.stage})`,
      new_percentage: result.newPercentage,
      stage: result.stage,
      latency_ms: latency
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to progress canary rollout',
      message: errorMsg
    };
    
  } finally {
    span.end();
  }
});

fastify.post('/canary/killswitch', async (request, reply) => {
  const span = LensTracer.createChildSpan('canary_killswitch');
  const startTime = Date.now();

  try {
    const { reason } = request.body as { reason?: string };
    const killReason = reason || 'Manual kill switch activation';
    
    globalFeatureFlags.killSwitchActivate(killReason);
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      reason: killReason,
    });

    return {
      success: true,
      message: 'Kill switch activated - all canary traffic stopped',
      reason: killReason,
      traffic_percentage: 0,
      latency_ms: latency
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to activate kill switch',
      message: errorMsg
    };
    
  } finally {
    span.end();
  }
});

// Phase D Quality Gates and Validation Endpoints
fastify.post('/validation/quality-gates', async (request, reply) => {
  const span = LensTracer.createChildSpan('quality_gates_run');
  const startTime = Date.now();

  try {
    const qualityReport = await runQualityGates();
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      overall_passed: qualityReport.overall_passed,
      gates_total: qualityReport.metrics_summary.gates_total,
      gates_passed: qualityReport.metrics_summary.gates_passed,
      promotion_eligible: qualityReport.promotion_eligible
    });

    return {
      success: true,
      quality_gates_report: qualityReport,
      latency_ms: latency
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Quality gates validation failed',
      message: errorMsg
    };
    
  } finally {
    span.end();
  }
});

fastify.post('/validation/nightly', async (request, reply) => {
  const span = LensTracer.createChildSpan('nightly_validation_run');
  const startTime = Date.now();

  try {
    const { duration_minutes, repo_types, languages, force_night } = request.body as any;
    
    const options = {
      duration_minutes: duration_minutes || 120,
      repo_types: repo_types || ['backend', 'frontend', 'monorepo'],
      languages: languages || ['typescript', 'javascript', 'python', 'go', 'rust'],
      force_night: force_night
    };

    const validationResult = await runNightlyValidation(options);
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      night: validationResult.night,
      validation_passed: validationResult.validation_passed,
      duration_minutes: validationResult.duration_minutes,
      blocking_issues: validationResult.blocking_issues.length
    });

    return {
      success: true,
      nightly_validation_result: validationResult,
      latency_ms: latency
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Nightly validation failed',
      message: errorMsg
    };
    
  } finally {
    span.end();
  }
});

fastify.get('/validation/status', async (request, reply) => {
  const span = LensTracer.createChildSpan('validation_status');
  const startTime = Date.now();

  try {
    const validationStatus = getValidationStatus();
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      current_night: validationStatus.current_night,
      consecutive_passes: validationStatus.consecutive_passes,
      promotion_ready: validationStatus.promotion_ready
    });

    return {
      success: true,
      validation_status: validationStatus,
      latency_ms: latency
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to get validation status',
      message: errorMsg
    };
    
  } finally {
    span.end();
  }
});

fastify.get('/validation/signoff-report', async (request, reply) => {
  const span = LensTracer.createChildSpan('signoff_report');
  const startTime = Date.now();

  try {
    const signoffReport = globalThreeNightValidation.generateSignoffReport();
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      promotion_ready: signoffReport.sign_off_report.promotion_ready
    });

    return {
      success: true,
      signoff_report: signoffReport,
      latency_ms: latency
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to generate signoff report',
      message: errorMsg
    };
    
  } finally {
    span.end();
  }
});

// Phase D Monitoring Dashboard Endpoint
fastify.get('/monitoring/dashboard', async (request, reply) => {
  const span = LensTracer.createChildSpan('dashboard_status');
  const startTime = Date.now();

  try {
    const dashboardState = getDashboardState();
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      health_status: dashboardState.health.status,
      active_alerts: dashboardState.health.active_alerts,
      canary_traffic: dashboardState.canary_status.traffic_percentage
    });

    return {
      success: true,
      dashboard_state: dashboardState,
      timestamp: new Date().toISOString(),
      latency_ms: latency
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to get dashboard state',
      message: errorMsg
    };
    
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
      'request.repo_sha': searchRequest.repo_sha,
      'request.query': searchRequest.q,
      'request.mode': searchRequest.mode,
      'request.fuzzy': searchRequest.fuzzy,
      'request.k': searchRequest.k,
      'trace_id': traceId,
    });

    // Perform search
    const result = await searchEngine.search({
      trace_id: traceId,
      repo_sha: searchRequest.repo_sha,
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
    const versionInfo = getVersionInfo();
    const response: SearchResponse = {
      hits: result.hits.map((hit: any) => ({
        file: hit.file,
        line: hit.line,
        col: hit.col,
        lang: hit.lang,
        snippet: hit.snippet,
        score: hit.score,
        why: hit.why,
        ast_path: hit.ast_path,
        symbol_kind: hit.symbol_kind,
        byte_offset: hit.byte_offset,
        span_len: hit.span_len,
        context_before: hit.context_before,
        context_after: hit.context_after,
      })),
      total: result.hits.length,
      latency_ms: {
        stage_a: result.stage_a_latency || 0,
        stage_b: result.stage_b_latency || 0,
        stage_c: result.stage_c_latency,
        total: totalLatency,
      },
      trace_id: traceId,
      api_version: versionInfo.api_version,
      index_version: versionInfo.index_version,
      policy_version: versionInfo.policy_version,
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

    // Handle INDEX_MISSING specifically
    if (errorMsg.includes('INDEX_MISSING')) {
      reply.status(503);
      return {
        error: 'INDEX_MISSING',
        message: 'Repository not found in index',
        hits: [],
        total: 0,
        latency_ms: {
          stage_a: 0,
          stage_b: 0,
          total: Date.now() - startTime,
        },
        trace_id: traceId,
        api_version: 'v1' as const,
        index_version: 'v1' as const,
        policy_version: 'v1' as const,
      };
    }

    // Return general error response
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
      api_version: 'v1' as const,
      index_version: 'v1' as const,
      policy_version: 'v1' as const,
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

    // Perform structural search using the main search method
    const result = await searchEngine.search({
      trace_id: traceId,
      repo_sha: structRequest.repo_sha,
      query: structRequest.pattern,
      mode: 'struct',
      k: structRequest.max_results || 100,
      fuzzy_distance: 0,
      started_at: new Date(),
      stages: [],
    });

    const totalLatency = Date.now() - startTime;

    // Check SLA compliance (30ms target)
    if (totalLatency > 30) {
      fastify.log.warn(`Struct search SLA breach: ${totalLatency}ms > 30ms`);
    }

    const response: SearchResponse = {
      hits: result.hits.map((hit: any) => ({
        file: hit.file,
        line: hit.line,
        col: hit.col,
        lang: hit.lang,
        snippet: hit.snippet,
        score: hit.score,
        why: hit.why,
        ast_path: hit.ast_path,
        symbol_kind: hit.symbol_kind,
        byte_offset: hit.byte_offset,
        span_len: hit.span_len,
        context_before: hit.context_before,
        context_after: hit.context_after,
      })),
      total: result.hits.length,
      latency_ms: {
        stage_a: result.stage_a_latency || 0,
        stage_b: result.stage_b_latency || 0,
        stage_c: result.stage_c_latency,
        total: totalLatency,
      },
      trace_id: traceId,
      api_version: 'v1' as const,
      index_version: 'v1' as const,
      policy_version: 'v1' as const,
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
      api_version: 'v1' as const,
      index_version: 'v1' as const,
      policy_version: 'v1' as const,
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

    // Find symbols near location - using search instead
    const result = await searchEngine.search({
      trace_id: traceId,
      repo_sha: 'lens-src', // Default repo for demo
      query: `file:${symbolsRequest.file}`,
      mode: 'struct',
      k: 20,
      fuzzy_distance: 0,
      started_at: new Date(),
      stages: [],
    });

    const totalLatency = Date.now() - startTime;

    // Check SLA compliance (15ms target)
    if (totalLatency > 15) {
      fastify.log.warn(`Symbols near SLA breach: ${totalLatency}ms > 15ms`);
    }

    const response: SearchResponse = {
      hits: result.hits.map((hit: any) => ({
        file: hit.file,
        line: hit.line,
        col: hit.col,
        lang: hit.lang,
        snippet: hit.snippet,
        score: hit.score,
        why: hit.why,
        ast_path: hit.ast_path,
        symbol_kind: hit.symbol_kind,
        byte_offset: hit.byte_offset,
        span_len: hit.span_len,
        context_before: hit.context_before,
        context_after: hit.context_after,
      })),
      total: result.hits.length,
      latency_ms: {
        stage_a: 0,
        stage_b: totalLatency, // Symbols are Stage B
        total: totalLatency,
      },
      trace_id: traceId,
      api_version: 'v1' as const,
      index_version: 'v1' as const,
      policy_version: 'v1' as const,
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
      api_version: 'v1' as const,
      index_version: 'v1' as const,
      policy_version: 'v1' as const,
    };

  } finally {
    span.end();
  }
});

// Phase 2 Enhancement: AST Coverage Statistics
fastify.get('/coverage/ast', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_coverage_ast');
  const startTime = Date.now();

  try {
    const coverageStats = searchEngine.getASTCoverageStats();
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      coverage_percentage: coverageStats.coverage.coveragePercentage,
      cached_ts_files: coverageStats.coverage.cachedTSFiles,
    });

    return {
      timestamp: new Date().toISOString(),
      ...coverageStats
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return { 
      error: 'Failed to get AST coverage stats',
      timestamp: new Date().toISOString()
    };
    
  } finally {
    span.end();
  }
});

// Phase 2 Enhancement: Learned Reranker Control (A/B Testing)
fastify.post('/reranker/enable', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_reranker_enable');
  const startTime = Date.now();

  try {
    const { enabled } = request.body as { enabled: boolean };
    
    if (typeof enabled !== 'boolean') {
      reply.status(400);
      return {
        success: false,
        error: 'enabled must be a boolean',
        enabled: false
      };
    }
    
    searchEngine.setRerankingEnabled(enabled);
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      reranker_enabled: enabled,
    });

    return {
      success: true,
      message: `Learned reranking ${enabled ? 'enabled' : 'disabled'}`,
      enabled,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to update reranker config',
      enabled: false,
      timestamp: new Date().toISOString()
    };
    
  } finally {
    span.end();
  }
});

// Phase B Enhancement: Stage-A Policy Configuration
fastify.patch('/policy/stageA', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_stage_a_config');
  const startTime = Date.now();

  try {
    const body = request.body as {
      rare_term_fuzzy?: boolean;
      synonyms_when_identifier_density_below?: number;
      prefilter?: { type: string; enabled: boolean };
      wand?: { enabled: boolean; block_max: boolean };
      per_file_span_cap?: number;
      native_scanner?: 'on' | 'off' | 'auto';
      k_candidates?: string | number; // "adaptive(180,380)" or fixed number
      fanout_features?: string; // "+rare_terms,+fuzzy_edits,+id_entropy,+path_var,+cand_slope" or "off"
    };
    
    // Validate configuration parameters
    if (body.synonyms_when_identifier_density_below !== undefined) {
      if (typeof body.synonyms_when_identifier_density_below !== 'number' || 
          body.synonyms_when_identifier_density_below < 0 || 
          body.synonyms_when_identifier_density_below > 1) {
        reply.status(400);
        return {
          success: false,
          error: 'synonyms_when_identifier_density_below must be a number between 0 and 1'
        };
      }
    }

    if (body.per_file_span_cap !== undefined) {
      if (typeof body.per_file_span_cap !== 'number' || 
          body.per_file_span_cap < 1 || 
          body.per_file_span_cap > 10) {
        reply.status(400);
        return {
          success: false,
          error: 'per_file_span_cap must be a number between 1 and 10'
        };
      }
    }

    if (body.native_scanner !== undefined && 
        !['on', 'off', 'auto'].includes(body.native_scanner)) {
      reply.status(400);
      return {
        success: false,
        error: 'native_scanner must be one of: "on", "off", "auto"'
      };
    }

    // Validate and configure adaptive fan-out
    let adaptiveEnabled = false;
    if (body.k_candidates !== undefined) {
      if (typeof body.k_candidates === 'string') {
        const adaptiveMatch = body.k_candidates.match(/^adaptive\((\d+),(\d+)\)$/);
        if (adaptiveMatch) {
          const [, min, max] = adaptiveMatch;
          const minVal = parseInt(min!, 10);
          const maxVal = parseInt(max!, 10);
          
          if (minVal < 100 || maxVal > 500 || minVal >= maxVal) {
            reply.status(400);
            return {
              success: false,
              error: 'adaptive k_candidates must be in format "adaptive(min,max)" with 100 <= min < max <= 500'
            };
          }
          
          globalAdaptiveFanout.updateConfig({
            k_candidates: { min: minVal, max: maxVal }
          });
          adaptiveEnabled = true;
        } else {
          reply.status(400);
          return {
            success: false,
            error: 'k_candidates string must be in format "adaptive(min,max)"'
          };
        }
      } else if (typeof body.k_candidates === 'number') {
        if (body.k_candidates < 100 || body.k_candidates > 500) {
          reply.status(400);
          return {
            success: false,
            error: 'k_candidates must be between 100 and 500'
          };
        }
      }
    }

    if (body.fanout_features !== undefined) {
      if (body.fanout_features === 'off') {
        adaptiveEnabled = false;
      } else if (body.fanout_features.startsWith('+')) {
        // Validate feature list format
        const features = body.fanout_features.substring(1).split(',');
        const validFeatures = ['rare_terms', 'fuzzy_edits', 'id_entropy', 'path_var', 'cand_slope'];
        
        for (const feature of features) {
          const cleanFeature = feature.replace(/^[+-]/, '');
          if (!validFeatures.includes(cleanFeature)) {
            reply.status(400);
            return {
              success: false,
              error: `Invalid fanout feature: ${cleanFeature}. Valid features: ${validFeatures.join(', ')}`
            };
          }
        }
        adaptiveEnabled = true;
      } else {
        reply.status(400);
        return {
          success: false,
          error: 'fanout_features must be "off" or "+feature1,+feature2,..." format'
        };
      }
    }

    // Enable/disable adaptive fanout based on configuration
    globalAdaptiveFanout.setEnabled(adaptiveEnabled);

    // Apply configuration to search engine (filter out undefined values)
    const stageAConfig: any = {};
    if (body.rare_term_fuzzy !== undefined) stageAConfig.rare_term_fuzzy = body.rare_term_fuzzy;
    if (body.synonyms_when_identifier_density_below !== undefined) stageAConfig.synonyms_when_identifier_density_below = body.synonyms_when_identifier_density_below;
    if (body.prefilter?.enabled !== undefined) stageAConfig.prefilter_enabled = body.prefilter.enabled;
    if (body.prefilter?.type !== undefined) stageAConfig.prefilter_type = body.prefilter.type;
    if (body.wand?.enabled !== undefined) stageAConfig.wand_enabled = body.wand.enabled;
    if (body.wand?.block_max !== undefined) stageAConfig.wand_block_max = body.wand.block_max;
    if (body.per_file_span_cap !== undefined) stageAConfig.per_file_span_cap = body.per_file_span_cap;
    if (body.native_scanner !== undefined) stageAConfig.native_scanner = body.native_scanner;
    if (body.k_candidates !== undefined) stageAConfig.k_candidates = body.k_candidates;
    if (body.fanout_features !== undefined) stageAConfig.fanout_features = body.fanout_features;
    stageAConfig.adaptive_enabled = adaptiveEnabled;
    
    await searchEngine.updateStageAConfig(stageAConfig);

    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      config_updated: JSON.stringify(body),
    });

    return {
      success: true,
      message: 'Stage-A configuration updated',
      applied_config: body,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to update Stage-A configuration',
      timestamp: new Date().toISOString()
    };
    
  } finally {
    span.end();
  }
});

// Phase B Enhancement: Enable/disable Phase B optimizations
fastify.post('/policy/phaseB/enable', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_phase_b_enable');
  const startTime = Date.now();

  try {
    const { enabled } = request.body as { enabled: boolean };
    
    if (typeof enabled !== 'boolean') {
      reply.status(400);
      return {
        success: false,
        error: 'enabled must be a boolean',
        enabled: false
      };
    }
    
    searchEngine.setPhaseBOptimizationsEnabled(enabled);
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      phase_b_enabled: enabled,
    });

    return {
      success: true,
      message: `Phase B optimizations ${enabled ? 'enabled' : 'disabled'}`,
      enabled,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to update Phase B config',
      enabled: false,
      timestamp: new Date().toISOString()
    };
    
  } finally {
    span.end();
  }
});

// Phase B Enhancement: Run benchmark suite
fastify.post('/bench/phaseB', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_phase_b_benchmark');
  const startTime = Date.now();

  try {
    const benchmarkResult = await searchEngine.runPhaseBBenchmark();
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      benchmark_status: benchmarkResult.overall_status,
      stage_a_p95_ms: benchmarkResult.stage_a_p95_ms,
    });

    return {
      success: true,
      message: 'Phase B benchmark completed',
      results: benchmarkResult,
      duration_ms: latency,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Phase B benchmark failed',
      message: errorMsg,
      timestamp: new Date().toISOString()
    };
    
  } finally {
    span.end();
  }
});

// Phase B Enhancement: Generate calibration plot data
fastify.get('/reports/calibration-plot', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_calibration_plot');
  const startTime = Date.now();

  try {
    const calibrationData = await searchEngine.generateCalibrationPlot();
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      calibration_error: calibrationData.calibration_error,
      reliability_score: calibrationData.reliability_score,
    });

    return {
      success: true,
      message: 'Calibration plot data generated',
      data: calibrationData,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Calibration plot generation failed',
      message: errorMsg,
      timestamp: new Date().toISOString()
    };
    
  } finally {
    span.end();
  }
});

// Phase 3 Enhancement: Semantic Stage Configuration
fastify.patch('/policy/stageC', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_semantic_config');
  const startTime = Date.now();

  try {
    const body = request.body as {
      gate?: { 
        nl_threshold?: number | string; // number or "adaptive(0.55→0.30)" 
        min_candidates?: number | string; // number or "adaptive(8→14)"
      };
      ann?: { 
        efSearch?: number | string; // number or "dynamic(...)" 
        k?: number;
        early_exit?: {
          after_probes?: number;
          margin_tau?: number;
          guards?: {
            require_symbol_or_struct?: boolean;
            min_top1_top5_margin?: number;
          };
        };
      };
      confidence_cutoff?: number;
    };
    
    // Validate configuration parameters
    let adaptiveGatesEnabled = false;
    
    if (body.gate?.nl_threshold !== undefined) {
      if (typeof body.gate.nl_threshold === 'string') {
        const adaptiveMatch = body.gate.nl_threshold.match(/^adaptive\(([\d.]+)→([\d.]+)\)$/);
        if (adaptiveMatch) {
          const [, max, min] = adaptiveMatch;
          const maxVal = parseFloat(max || '0');
          const minVal = parseFloat(min || '0');
          
          if (maxVal < 0 || maxVal > 1 || minVal < 0 || minVal > 1 || minVal >= maxVal) {
            reply.status(400);
            return {
              success: false,
              error: 'adaptive nl_threshold must be in format "adaptive(max→min)" with 0 <= min < max <= 1'
            };
          }
          
          globalAdaptiveFanout.updateConfig({
            gate: {
              ...globalAdaptiveFanout['config'].gate,
              nl_threshold: { min: minVal, max: maxVal }
            }
          });
          adaptiveGatesEnabled = true;
        } else {
          reply.status(400);
          return {
            success: false,
            error: 'nl_threshold string must be in format "adaptive(max→min)"'
          };
        }
      } else if (typeof body.gate.nl_threshold === 'number') {
        if (body.gate.nl_threshold < 0 || body.gate.nl_threshold > 1) {
          reply.status(400);
          return {
            success: false,
            error: 'nl_threshold must be a number between 0 and 1'
          };
        }
      }
    }

    if (body.gate?.min_candidates !== undefined) {
      if (typeof body.gate.min_candidates === 'string') {
        const adaptiveMatch = body.gate.min_candidates.match(/^adaptive\((\d+)→(\d+)\)$/);
        if (adaptiveMatch) {
          const [, min, max] = adaptiveMatch;
          const minVal = parseInt(min!, 10);
          const maxVal = parseInt(max!, 10);
          
          if (minVal < 5 || maxVal > 50 || minVal >= maxVal) {
            reply.status(400);
            return {
              success: false,
              error: 'adaptive min_candidates must be in format "adaptive(min→max)" with 5 <= min < max <= 50'
            };
          }
          
          globalAdaptiveFanout.updateConfig({
            gate: {
              ...globalAdaptiveFanout['config'].gate,
              min_candidates: { min: minVal, max: maxVal }
            }
          });
          adaptiveGatesEnabled = true;
        } else {
          reply.status(400);
          return {
            success: false,
            error: 'min_candidates string must be in format "adaptive(min→max)"'
          };
        }
      } else if (typeof body.gate.min_candidates === 'number') {
        if (body.gate.min_candidates < 10 || body.gate.min_candidates > 500) {
          reply.status(400);
          return {
            success: false,
            error: 'min_candidates must be a number between 10 and 500'
          };
        }
      }
    }

    if (body.ann?.efSearch !== undefined) {
      if (typeof body.ann.efSearch === 'string') {
        // Validate dynamic efSearch formula: "dynamic(48 + 24*log2(1 + |candidates|/150))"
        const dynamicMatch = body.ann.efSearch.match(/^dynamic\((.+)\)$/);
        if (!dynamicMatch) {
          reply.status(400);
          return {
            success: false,
            error: 'efSearch string must be in format "dynamic(formula)"'
          };
        }
        // Formula validation is deferred to runtime
      } else if (typeof body.ann.efSearch === 'number') {
        if (body.ann.efSearch < 16 || body.ann.efSearch > 512) {
          reply.status(400);
          return {
            success: false,
            error: 'efSearch must be a number between 16 and 512'
          };
        }
      }
    }

    // Validate early exit parameters
    if (body.ann?.early_exit) {
      const earlyExit = body.ann.early_exit;
      
      if (earlyExit.after_probes !== undefined && (earlyExit.after_probes < 16 || earlyExit.after_probes > 256)) {
        reply.status(400);
        return {
          success: false,
          error: 'early_exit.after_probes must be between 16 and 256'
        };
      }
      
      if (earlyExit.margin_tau !== undefined && (earlyExit.margin_tau < 0.01 || earlyExit.margin_tau > 0.5)) {
        reply.status(400);
        return {
          success: false,
          error: 'early_exit.margin_tau must be between 0.01 and 0.5'
        };
      }

      if (earlyExit.guards?.min_top1_top5_margin !== undefined && 
          (earlyExit.guards.min_top1_top5_margin < 0.05 || earlyExit.guards.min_top1_top5_margin > 0.5)) {
        reply.status(400);
        return {
          success: false,
          error: 'early_exit.guards.min_top1_top5_margin must be between 0.05 and 0.5'
        };
      }
    }

    // Apply configuration to search engine
    const semanticConfig: any = {
      adaptive_gates_enabled: adaptiveGatesEnabled,
    };
    
    if (body.gate?.nl_threshold != null) semanticConfig.nl_threshold = body.gate.nl_threshold;
    if (body.gate?.min_candidates != null || body.ann?.k != null) {
      semanticConfig.min_candidates = body.gate?.min_candidates ?? body.ann?.k;
    }
    if (body.ann?.efSearch != null) semanticConfig.efSearch = body.ann.efSearch;
    if (body.confidence_cutoff != null) semanticConfig.confidence_cutoff = body.confidence_cutoff;
    if (body.ann?.k != null) semanticConfig.ann_k = body.ann.k;
    if (body.ann?.early_exit != null) semanticConfig.early_exit = body.ann.early_exit;
    
    await searchEngine.updateSemanticConfig(semanticConfig);

    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      config_updated: JSON.stringify(body),
    });

    return {
      success: true,
      message: 'Semantic stage configuration updated',
      applied_config: body,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to update semantic configuration',
      timestamp: new Date().toISOString()
    };
    
  } finally {
    span.end();
  }
});

// Policy dump endpoint - Phase 1 baseline snapshot requirement
fastify.get('/policy/dump', async (request, reply) => {
  const span = LensTracer.createChildSpan('policy_dump');
  const startTime = Date.now();

  try {
    // Gather complete system policy configuration for baseline snapshot
    const policyDump = {
      api_version: getVersionInfo().api_version,
      index_version: getVersionInfo().index_version,
      policy_version: getVersionInfo().policy_version,
      stage_configurations: {
        stage_a: {
          rare_term_fuzzy: true,
          synonyms_when_identifier_density_below: 0.3,
          prefilter: { type: 'bigram', enabled: true },
          wand: { enabled: true, block_max: true },
          per_file_span_cap: 3,
          native_scanner: 'auto'
        },
        stage_b: {
          enabled: globalFeatureFlags.getCanaryStatus().stageFlags.stageB_enabled,
          symbol_ranking_enabled: true,
          reranker_enabled: true,
          max_candidates: 200
        },
        stage_c: {
          enabled: globalFeatureFlags.getCanaryStatus().stageFlags.stageC_enabled,
          semantic_gating: {
            nl_likelihood_threshold: 0.5,
            min_candidates: 10
          },
          ann_config: {
            efSearch: 64,
            k: 50
          },
          confidence_cutoff: 0.1
        }
      },
      kill_switches: {
        stage_b_enabled: globalFeatureFlags.getCanaryStatus().stageFlags.stageB_enabled,
        stage_c_enabled: globalFeatureFlags.getCanaryStatus().stageFlags.stageC_enabled,
        stage_a_native_scanner: globalFeatureFlags.getCanaryStatus().stageFlags.stageA_native_scanner,
        kill_switch_active: globalFeatureFlags.getCanaryStatus().killSwitchEnabled
      },
      telemetry: {
        trace_sample_rate: 0.1,
        metrics_enabled: true
      },
      quality_gates: {
        ndcg_improvement_threshold: 0.02,
        recall_maintenance_threshold: 0.85,
        latency_increase_threshold: 0.10
      },
      timestamp: new Date().toISOString(),
      config_fingerprint: `baseline-${Date.now()}`
    };

    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      stage_a_native_scanner: policyDump.stage_configurations.stage_a.native_scanner,
      stage_b_enabled: policyDump.stage_configurations.stage_b.enabled,
      stage_c_enabled: policyDump.stage_configurations.stage_c.enabled
    });

    return policyDump;

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      error: 'Failed to dump policy configuration',
      message: errorMsg,
      timestamp: new Date().toISOString()
    };
    
  } finally {
    span.end();
  }
});

// Phase 2 Enhancement: Recall Pack Orchestration
fastify.post('/phase2/execute', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_phase2_execute');
  const startTime = Date.now();

  try {
    // Import Phase 2 orchestrator
    const { Phase2RecallPack } = await import('../core/phase2-recall-pack.js');
    
    const body = request.body as {
      index_root?: string;
      output_dir?: string;
      api_base_url?: string;
    };
    
    const phase2 = new Phase2RecallPack(
      body.index_root || './indexed-content',
      body.output_dir || './phase2-results',
      body.api_base_url || 'http://localhost:3001'
    );
    
    console.log('🎯 Starting Phase 2 Recall Pack execution via API...');
    
    const results = await phase2.executePhase2();
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      recall_improvement_pct: results.recall_improvement_pct,
      promotion_ready: results.promotion_ready,
    });

    return {
      success: true,
      message: 'Phase 2 Recall Pack execution completed',
      results,
      duration_ms: latency,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Phase 2 execution failed',
      message: errorMsg,
      timestamp: new Date().toISOString()
    };
    
  } finally {
    span.end();
  }
});

// Phase 2 Enhancement: Mine Synonyms
fastify.post('/phase2/synonyms/mine', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_phase2_mine_synonyms');
  const startTime = Date.now();

  try {
    const { Phase2SynonymMiner } = await import('../core/phase2-synonym-miner.js');
    
    const body = request.body as {
      tau_pmi?: number;
      min_freq?: number;
      k_synonyms?: number;
      index_root?: string;
      output_dir?: string;
    };
    
    const synonymMiner = new Phase2SynonymMiner(
      body.index_root || './indexed-content',
      body.output_dir || './synonyms'
    );
    
    const synonymTable = await synonymMiner.mineSynonyms({
      tau_pmi: body.tau_pmi || 3.0,
      min_freq: body.min_freq || 20,
      k_synonyms: body.k_synonyms || 8,
    });
    
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      synonym_entries: synonymTable.entries.length,
    });

    return {
      success: true,
      message: 'Synonym mining completed',
      synonym_table: synonymTable,
      duration_ms: latency,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Synonym mining failed',
      message: errorMsg,
      timestamp: new Date().toISOString()
    };
    
  } finally {
    span.end();
  }
});

// Phase 2 Enhancement: Refit Path Prior
fastify.post('/phase2/pathprior/refit', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_phase2_refit_pathprior');
  const startTime = Date.now();

  try {
    const { Phase2PathPrior } = await import('../core/phase2-path-prior.js');
    
    const body = request.body as {
      l2_regularization?: number;
      debias_low_priority_paths?: boolean;
      max_deboost?: number;
      index_root?: string;
      output_dir?: string;
    };
    
    const pathPrior = new Phase2PathPrior(
      body.index_root || './indexed-content',
      body.output_dir || './path-priors'
    );
    
    const model = await pathPrior.refitPathPrior({
      l2_regularization: body.l2_regularization || 1.0,
      debias_low_priority_paths: body.debias_low_priority_paths !== false,
      max_deboost: body.max_deboost || 0.6,
    });
    
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      model_version: model.version,
      auc_roc: model.performance.auc_roc,
    });

    return {
      success: true,
      message: 'Path prior refitting completed',
      model,
      duration_ms: latency,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Path prior refitting failed',
      message: errorMsg,
      timestamp: new Date().toISOString()
    };
    
  } finally {
    span.end();
  }
});

//
// ==== Phase 3 - Precision/Semantic Pack Endpoints ====
//

// Phase 3 Main Execution
fastify.post('/phase3/execute', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_phase3_execute');
  const startTime = Date.now();

  try {
    // Import Phase 3 orchestrator
    const { Phase3PrecisionPack } = await import('../core/phase3-precision-pack.js');
    
    const body = request.body as {
      index_root?: string;
      output_dir?: string;
      api_base_url?: string;
      config?: any;
    };
    
    const phase3 = new Phase3PrecisionPack(
      body.index_root || './indexed-content',
      body.output_dir || './phase3-results',
      body.api_base_url || 'http://localhost:3001'
    );
    
    console.log('🎯 Starting Phase 3 Precision/Semantic Pack execution via API...');
    
    const results = await phase3.execute(body.config);
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      ndcg_improvement_points: results.ndcg_improvement_points,
      promotion_ready: results.promotion_ready,
    });

    return {
      success: true,
      message: 'Phase 3 Precision/Semantic Pack execution completed',
      results,
      duration_ms: latency,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Phase 3 execution failed',
      message: errorMsg,
      timestamp: new Date().toISOString()
    };
    
  } finally {
    span.end();
  }
});

// Phase 3 Pattern Pack Management
fastify.post('/phase3/patterns/find', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_phase3_find_patterns');
  const startTime = Date.now();

  try {
    const { Phase3PatternPackEngine } = await import('../core/phase3-pattern-packs.js');
    
    const body = request.body as {
      source_code: string;
      file_path: string;
      language: string;
      pattern_names?: string[];
    };
    
    if (!body.source_code || !body.file_path || !body.language) {
      reply.status(400);
      return {
        success: false,
        error: 'Missing required fields: source_code, file_path, language',
        timestamp: new Date().toISOString()
      };
    }
    
    const engine = new Phase3PatternPackEngine();
    const patterns = await engine.findPatterns(
      body.source_code,
      body.file_path,
      body.language,
      body.pattern_names
    );
    
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      patterns_found: patterns.length,
      file_path: body.file_path,
      language: body.language,
    });

    return {
      success: true,
      message: 'Pattern matching completed',
      patterns,
      duration_ms: latency,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Pattern matching failed',
      message: errorMsg,
      timestamp: new Date().toISOString()
    };
    
  } finally {
    span.end();
  }
});

// Phase 3 Configuration and Status
fastify.get('/phase3/config', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_phase3_get_config');

  try {
    const { Phase3PrecisionPack } = await import('../core/phase3-precision-pack.js');
    const { Phase3PatternPackEngine } = await import('../core/phase3-pattern-packs.js');
    
    const phase3 = new Phase3PrecisionPack();
    const engine = new Phase3PatternPackEngine();
    
    const config = phase3.getDefaultConfig();
    const acceptanceGates = phase3.getAcceptanceGates();
    const tripwireChecks = phase3.getTripwireChecks();
    const patternStats = engine.getStatistics();

    span.setAttributes({
      success: true,
      pattern_packs_count: patternStats.total_packs,
      total_patterns: patternStats.total_patterns,
      languages_supported: patternStats.languages_supported.length,
    });

    return {
      success: true,
      config,
      acceptance_gates: acceptanceGates,
      tripwire_checks: tripwireChecks,
      pattern_statistics: patternStats,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to retrieve Phase 3 configuration',
      message: errorMsg,
      timestamp: new Date().toISOString()
    };
    
  } finally {
    span.end();
  }
});

// Phase 3 Rollback 
fastify.post('/phase3/rollback', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_phase3_rollback');
  const startTime = Date.now();

  try {
    const { Phase3PrecisionPack } = await import('../core/phase3-precision-pack.js');
    
    const phase3 = new Phase3PrecisionPack();
    
    console.log('🔄 Performing Phase 3 rollback via API...');
    
    await phase3.performRollback();
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
    });

    return {
      success: true,
      message: 'Phase 3 rollback completed successfully',
      duration_ms: latency,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Phase 3 rollback failed',
      message: errorMsg,
      timestamp: new Date().toISOString()
    };
    
  } finally {
    span.end();
  }
});

//
// ==== Precision Optimization Pipeline Endpoints ====
//

// Block A configuration is handled by the existing /policy/stageC route above

// Apply Block B: Calibrated dynamic_topn
fastify.patch('/policy/output', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_precision_block_b');
  const startTime = Date.now();

  try {
    const body = request.body as {
      dynamic_topn?: {
        enabled: boolean;
        score_threshold: number;
        hard_cap: number;
      };
    };

    // Validate Block B configuration
    if (body.dynamic_topn) {
      if (body.dynamic_topn.score_threshold < 0 || body.dynamic_topn.score_threshold > 1) {
        reply.status(400);
        return {
          success: false,
          error: 'dynamic_topn.score_threshold must be between 0 and 1'
        };
      }
      if (body.dynamic_topn.hard_cap < 5 || body.dynamic_topn.hard_cap > 50) {
        reply.status(400);
        return {
          success: false,
          error: 'dynamic_topn.hard_cap must be between 5 and 50'
        };
      }
    }

    // Enable Block B and apply configuration
    searchEngine.setPrecisionOptimizationEnabled('B', true);

    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      block_b_enabled: true,
      dynamic_topn_enabled: body.dynamic_topn?.enabled || false
    });

    return {
      success: true,
      message: 'Block B calibrated dynamic_topn applied',
      config: body,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to apply Block B configuration',
      message: errorMsg
    };
  } finally {
    span.end();
  }
});

// Apply Block C: Gentle deduplication
fastify.patch('/policy/precision', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_precision_block_c');
  const startTime = Date.now();

  try {
    const body = request.body as {
      dedup?: {
        in_file?: {
          simhash?: {
            k: number;
            hamming_max: number;
          };
          keep: number;
        };
        cross_file?: {
          vendor_deboost: number;
        };
      };
    };

    // Validate Block C configuration
    if (body.dedup?.in_file?.simhash) {
      if (body.dedup.in_file.simhash.k < 3 || body.dedup.in_file.simhash.k > 10) {
        reply.status(400);
        return {
          success: false,
          error: 'dedup.in_file.simhash.k must be between 3 and 10'
        };
      }
      if (body.dedup.in_file.simhash.hamming_max < 1 || body.dedup.in_file.simhash.hamming_max > 5) {
        reply.status(400);
        return {
          success: false,
          error: 'dedup.in_file.simhash.hamming_max must be between 1 and 5'
        };
      }
    }

    if (body.dedup?.in_file?.keep) {
      if (body.dedup.in_file.keep < 1 || body.dedup.in_file.keep > 10) {
        reply.status(400);
        return {
          success: false,
          error: 'dedup.in_file.keep must be between 1 and 10'
        };
      }
    }

    if (body.dedup?.cross_file?.vendor_deboost) {
      if (body.dedup.cross_file.vendor_deboost < 0 || body.dedup.cross_file.vendor_deboost > 1) {
        reply.status(400);
        return {
          success: false,
          error: 'dedup.cross_file.vendor_deboost must be between 0 and 1'
        };
      }
    }

    // Enable Block C and apply configuration
    searchEngine.setPrecisionOptimizationEnabled('C', true);

    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      block_c_enabled: true,
      simhash_k: body.dedup?.in_file?.simhash?.k || 5
    });

    return {
      success: true,
      message: 'Block C gentle deduplication applied',
      config: body,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to apply Block C configuration',
      message: errorMsg
    };
  } finally {
    span.end();
  }
});

// A/B Experiment Management Endpoints

// Create a new precision optimization experiment
fastify.post('/experiments/precision', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_create_precision_experiment');
  const startTime = Date.now();

  try {
    const body = request.body as {
      experiment_id: string;
      name: string;
      description?: string;
      traffic_percentage: number;
      treatment_config: any;
      promotion_gates: {
        min_ndcg_improvement_pct: number;
        min_recall_at_50: number;
        min_span_coverage_pct: number;
        max_latency_multiplier: number;
      };
    };

    // Validate experiment configuration
    if (!body.experiment_id || !body.name) {
      reply.status(400);
      return {
        success: false,
        error: 'experiment_id and name are required'
      };
    }

    if (body.traffic_percentage < 0 || body.traffic_percentage > 100) {
      reply.status(400);
      return {
        success: false,
        error: 'traffic_percentage must be between 0 and 100'
      };
    }

    // Create experiment configuration
    const experimentConfig = {
      experiment_id: body.experiment_id,
      name: body.name,
      description: body.description,
      traffic_percentage: body.traffic_percentage,
      treatment_config: body.treatment_config,
      promotion_gates: body.promotion_gates,
      anchor_validation_required: true,
      ladder_validation_required: true
    };

    await globalExperimentFramework.createExperiment(experimentConfig);

    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      experiment_id: body.experiment_id,
      traffic_percentage: body.traffic_percentage
    });

    return {
      success: true,
      message: 'Precision optimization experiment created',
      experiment_id: body.experiment_id,
      config: experimentConfig,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to create precision experiment',
      message: errorMsg
    };
  } finally {
    span.end();
  }
});

// Run Anchor validation for an experiment
fastify.post('/experiments/precision/:experimentId/validate/anchor', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_anchor_validation');
  const startTime = Date.now();

  try {
    const { experimentId } = request.params as { experimentId: string };

    const validationResult = await globalExperimentFramework.runAnchorValidation(experimentId);
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      experiment_id: experimentId,
      validation_passed: validationResult.passed
    });

    return {
      success: true,
      message: 'Anchor validation completed',
      experiment_id: experimentId,
      validation_result: validationResult,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Anchor validation failed',
      message: errorMsg
    };
  } finally {
    span.end();
  }
});

// Run Ladder validation for an experiment
fastify.post('/experiments/precision/:experimentId/validate/ladder', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_ladder_validation');
  const startTime = Date.now();

  try {
    const { experimentId } = request.params as { experimentId: string };

    const validationResult = await globalExperimentFramework.runLadderValidation(experimentId);
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      experiment_id: experimentId,
      validation_passed: validationResult.passed
    });

    return {
      success: true,
      message: 'Ladder validation completed',
      experiment_id: experimentId,
      validation_result: validationResult,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Ladder validation failed',
      message: errorMsg
    };
  } finally {
    span.end();
  }
});

// Check experiment promotion readiness
fastify.get('/experiments/precision/:experimentId/promotion', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_check_promotion');
  const startTime = Date.now();

  try {
    const { experimentId } = request.params as { experimentId: string };

    const promotionStatus = await globalExperimentFramework.checkPromotionReadiness(experimentId);
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      experiment_id: experimentId,
      promotion_ready: promotionStatus.ready
    });

    return {
      success: true,
      message: 'Promotion readiness checked',
      experiment_id: experimentId,
      promotion_status: promotionStatus,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to check promotion readiness',
      message: errorMsg
    };
  } finally {
    span.end();
  }
});

// Rollback an experiment
fastify.post('/experiments/precision/:experimentId/rollback', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_experiment_rollback');
  const startTime = Date.now();

  try {
    const { experimentId } = request.params as { experimentId: string };

    await globalExperimentFramework.rollbackExperiment(experimentId);
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      experiment_id: experimentId
    });

    return {
      success: true,
      message: 'Experiment rolled back successfully',
      experiment_id: experimentId,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to rollback experiment',
      message: errorMsg
    };
  } finally {
    span.end();
  }
});

// Get experiment status and results
fastify.get('/experiments/precision/:experimentId', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_get_experiment_status');
  const startTime = Date.now();

  try {
    const { experimentId } = request.params as { experimentId: string };

    const status = globalExperimentFramework.getExperimentStatus(experimentId);
    const latency = Date.now() - startTime;

    if (!status.config) {
      reply.status(404);
      return {
        success: false,
        error: 'Experiment not found',
        experiment_id: experimentId
      };
    }

    span.setAttributes({
      success: true,
      latency_ms: latency,
      experiment_id: experimentId,
      results_count: status.results.length
    });

    return {
      success: true,
      message: 'Experiment status retrieved',
      experiment_id: experimentId,
      status,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to get experiment status',
      message: errorMsg
    };
  } finally {
    span.end();
  }
});

// Utility functions for SPI
function generateStableRef(repoSha: string, filePath: string, sourceHash: string, lineStart: number, lineEnd: number, byteStart: number, byteEnd: number, astPath?: string): string {
  let ref = `lens://${repoSha}/${filePath}@${sourceHash}#L${lineStart}:${lineEnd}|B${byteStart}:${byteEnd}`;
  if (astPath) {
    ref += `|AST:${astPath}`;
  }
  return ref;
}

function deterministicStringify(obj: any): string {
  return JSON.stringify(obj, Object.keys(obj).sort());
}

function estimateTokenCount(text: string): number {
  // Rough estimation: 1 token ≈ 4 characters (conservative for code/technical content)
  // This is a simplified approach - real implementations might use tiktoken or similar
  return Math.ceil(text.length / 4);
}

function estimateObjectTokens(obj: any): number {
  const jsonStr = JSON.stringify(obj);
  return estimateTokenCount(jsonStr);
}

function calculateSourceHash(content: string): string {
  // Simple hash function for content integrity
  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return Math.abs(hash).toString(16);
}

// ==== SEARCH SPI ENDPOINTS ====

// Phase 0: SPI Search endpoint (facade over existing /search)
fastify.post('/v1/spi/search', async (request, reply): Promise<SpiSearchResponse> => {
  const span = LensTracer.createChildSpan('spi_search');
  const startTime = Date.now();
  const traceId = request.headers['x-trace-id'] as string || uuidv4();

  try {
    // Validate SPI request
    const spiRequest = SpiSearchRequestSchema.parse(request.body);
    
    // Check budget timeout early
    const budgetMs = spiRequest.budget_ms || 2000;
    const timeoutId = setTimeout(() => {
      // Budget exceeded - will be caught by timeout handler
    }, budgetMs);

    // Convert SPI request to internal search request
    const searchRequest: SearchRequest = {
      repo_sha: spiRequest.repo_sha,
      q: spiRequest.q,
      mode: spiRequest.mode,
      fuzzy: spiRequest.fuzzy,
      k: spiRequest.k,
      timeout_ms: budgetMs,
    };

    span.setAttributes({
      'request.repo_sha': spiRequest.repo_sha,
      'request.query': spiRequest.q,
      'request.mode': spiRequest.mode,
      'request.budget_ms': budgetMs,
      'trace_id': traceId,
    });

    // Perform search using existing engine
    const result = await searchEngine.search({
      trace_id: traceId,
      repo_sha: spiRequest.repo_sha,
      query: spiRequest.q,
      mode: spiRequest.mode,
      k: spiRequest.k,
      fuzzy_distance: spiRequest.fuzzy,
      started_at: new Date(),
      stages: [],
    });

    clearTimeout(timeoutId);
    const totalLatency = Date.now() - startTime;
    const timedOut = totalLatency > budgetMs;

    // Transform hits to SPI format with stable refs (Phase 1 preparation)
    const allSpiHits = result.hits.map((hit: any, index: number) => {
      const sourceHash = calculateSourceHash(hit.snippet || '');
      const byteStart = hit.byte_offset || 0;
      const byteEnd = byteStart + (hit.span_len || hit.snippet?.length || 0);
      
      const spiHit: any = {
        ...hit,
        source_hash: sourceHash,
        byte_start: byteStart,
        byte_end: byteEnd,
        ref: generateStableRef(
          spiRequest.repo_sha,
          hit.file,
          sourceHash,
          hit.line,
          hit.line + (hit.snippet?.split('\n').length || 1) - 1,
          byteStart,
          byteEnd,
          hit.ast_path
        )
      };

      return spiHit;
    });

    // Sort deterministically for reproducibility
    allSpiHits.sort((a, b) => {
      if (a.score !== b.score) return b.score - a.score; // Higher scores first
      if (a.file !== b.file) return a.file.localeCompare(b.file);
      return a.line - b.line;
    });

    // Apply token budget and pagination
    const tokenBudget = spiRequest.token_budget || 10000;
    const page = spiRequest.page || 0;
    const spiHits: any[] = [];
    let currentTokens = 0;
    let budgetExceeded = false;
    let hitsPerPage = 0;
    
    // Calculate hits per page by estimating tokens for first few results
    if (allSpiHits.length > 0 && hitsPerPage === 0) {
      // Estimate tokens per hit based on first 3 hits (or all if less than 3)
      const sampleSize = Math.min(3, allSpiHits.length);
      let sampleTokens = 0;
      for (let i = 0; i < sampleSize; i++) {
        sampleTokens += estimateObjectTokens(allSpiHits[i]);
      }
      const avgTokensPerHit = Math.ceil(sampleTokens / sampleSize);
      hitsPerPage = Math.max(1, Math.floor(tokenBudget / avgTokensPerHit));
    }

    // Calculate pagination window
    const startIndex = page * hitsPerPage;
    const endIndex = startIndex + hitsPerPage;
    const pageHits = allSpiHits.slice(startIndex, endIndex);

    // Add hits while respecting token budget
    for (const hit of pageHits) {
      const hitTokens = estimateObjectTokens(hit);
      if (currentTokens + hitTokens > tokenBudget) {
        budgetExceeded = true;
        break;
      }
      spiHits.push(hit);
      currentTokens += hitTokens;
    }

    // Calculate pagination metadata
    const totalPages = Math.ceil(allSpiHits.length / hitsPerPage);
    const hasNextPage = page < totalPages - 1;
    const hasPrevPage = page > 0;

    // Build SPI response
    const versionInfo = getVersionInfo();
    const response: SpiSearchResponse = {
      hits: spiHits,
      total: allSpiHits.length, // Total available hits across all pages
      latency_ms: {
        stage_a: result.stage_a_latency || 0,
        stage_b: result.stage_b_latency || 0,
        stage_c: result.stage_c_latency,
        total: totalLatency,
      },
      trace_id: traceId,
      api_version: versionInfo.api_version,
      index_version: versionInfo.index_version,
      policy_version: versionInfo.policy_version,
      timed_out: timedOut,
      // Token budget management
      token_usage: {
        used_tokens: currentTokens,
        budget_tokens: tokenBudget,
        budget_exceeded: budgetExceeded,
        estimated_total_tokens: currentTokens,
      },
      // Pagination metadata
      pagination: {
        page: page,
        results_in_page: spiHits.length,
        total_results: totalPages * hitsPerPage,
        has_next_page: hasNextPage,
        next_page: hasNextPage ? page + 1 : undefined,
        budget_per_page: tokenBudget,
      },
    };

    // Validate SPI response
    SpiSearchResponseSchema.parse(response);

    span.setAttributes({
      success: true,
      hits_count: response.hits.length,
      total_latency_ms: totalLatency,
      timed_out: timedOut,
    });

    reply.header('content-type', 'application/json; charset=utf-8');
    return JSON.parse(deterministicStringify(response));

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    // Handle timeout specifically
    if (errorMsg.includes('timeout') || Date.now() - startTime > (request.body as any)?.budget_ms) {
      const versionInfo = getVersionInfo();
      return {
        hits: [],
        total: 0,
        latency_ms: {
          stage_a: 0,
          stage_b: 0,
          total: Date.now() - startTime,
        },
        trace_id: traceId,
        api_version: versionInfo.api_version,
        index_version: versionInfo.index_version,
        policy_version: versionInfo.policy_version,
        timed_out: true,
      };
    }

    reply.status(400);
    throw error;

  } finally {
    span.end();
  }
});

// Phase 0: SPI Health endpoint (enhanced with SLA data)
fastify.get('/v1/spi/health', async (request, reply): Promise<SpiHealthResponse> => {
  const span = LensTracer.createChildSpan('spi_health');
  const startTime = Date.now();

  try {
    const health = await searchEngine.getHealthStatus();
    const latency = Date.now() - startTime;
    const versionInfo = getVersionInfo();

    const response: SpiHealthResponse = {
      status: health.status,
      timestamp: new Date().toISOString(),
      shards_healthy: health.shards_healthy,
      sla: {
        p95_latency_ms: 18,  // Mock SLA data - would come from metrics
        p99_latency_ms: 28,
        availability_pct: 99.9,
        error_rate_pct: 0.1,
      },
      version_info: {
        api_version: versionInfo.api_version,
        index_version: versionInfo.index_version,
        policy_version: versionInfo.policy_version,
      },
    };

    SpiHealthResponseSchema.parse(response);
    
    span.setAttributes({
      success: true,
      latency_ms: latency,
      status: health.status,
    });

    reply.header('content-type', 'application/json; charset=utf-8');
    return JSON.parse(deterministicStringify(response));

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(503);
    throw error;
    
  } finally {
    span.end();
  }
});

// Phase 1: Resolve endpoint
fastify.get('/v1/spi/resolve', async (request, reply): Promise<ResolveResponse> => {
  const span = LensTracer.createChildSpan('spi_resolve');
  const startTime = Date.now();

  try {
    const { ref } = request.query as { ref: string };
    const resolveRequest = ResolveRequestSchema.parse({ ref });

    // Parse the lens:// URI
    const refMatch = ref.match(/^lens:\/\/([^\/]+)\/([^@]+)@([^#]+)#L(\d+):(\d+)\|B(\d+):(\d+)(?:\|AST:(.+))?$/);
    if (!refMatch) {
      reply.status(400);
      throw new Error('Invalid lens reference format');
    }

    const [, repoSha, filePath, sourceHash, lineStart, lineEnd, byteStart, byteEnd, astPath] = refMatch;

    // Mock implementation - in real version would read from index
    const mockContent = `// Mock resolved content for ${filePath}
function example() {
  return "This is line ${lineStart}";
}`;

    const response: ResolveResponse = {
      ref,
      file_path: filePath,
      content: mockContent,
      source_hash: sourceHash,
      line_start: parseInt(lineStart),
      line_end: parseInt(lineEnd),
      byte_start: parseInt(byteStart),
      byte_end: parseInt(byteEnd),
      ast_path: astPath,
      surrounding_lines: {
        before: ['// Line before'],
        after: ['// Line after'],
      },
      metadata: {
        lang: 'typescript',
        symbol_kind: 'function',
        symbol_name: 'example',
      },
    };

    ResolveResponseSchema.parse(response);
    
    span.setAttributes({
      success: true,
      latency_ms: Date.now() - startTime,
      file_path: filePath,
    });

    reply.header('content-type', 'application/json; charset=utf-8');
    return JSON.parse(deterministicStringify(response));

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(400);
    throw error;
    
  } finally {
    span.end();
  }
});

// Phase 2: Context endpoint (batch resolve)
fastify.post('/v1/spi/context', async (request, reply): Promise<ContextResponse> => {
  const span = LensTracer.createChildSpan('spi_context');
  const startTime = Date.now();

  try {
    const contextRequest = ContextRequestSchema.parse(request.body);
    const tokenBudget = contextRequest.token_budget || 10000;
    let totalTokens = 0;
    const contexts = [];
    let budgetExceeded = false;

    for (const ref of contextRequest.refs) {
      if (totalTokens >= tokenBudget) {
        budgetExceeded = true;
        break;
      }
      
      // Mock content resolution - would resolve actual lens:// refs in production
      const content = `// Resolved context for ${ref}\nfunction mockFunction() {\n  return "content";\n}`;
      const tokenCount = estimateTokenCount(content);
      
      if (totalTokens + tokenCount <= tokenBudget) {
        contexts.push({
          ref,
          content,
          token_count: tokenCount,
          truncated: false,
        });
        totalTokens += tokenCount;
      } else {
        // Truncate to fit budget
        const remainingTokens = tokenBudget - totalTokens;
        if (remainingTokens > 50) { // Only include if we have meaningful space
          const truncatedContent = content.substring(0, remainingTokens * 4);
          contexts.push({
            ref,
            content: truncatedContent,
            token_count: remainingTokens,
            truncated: true,
          });
          totalTokens = tokenBudget;
        }
        budgetExceeded = true;
        break;
      }
    }

    const response: ContextResponse = {
      contexts,
      total_tokens: totalTokens,
      deduped_count: 0, // Mock - would implement deduplication
      // budget_exceeded not supported in ContextResponse schema
      // refs_omitted: contextRequest.refs.length - contexts.length,
    };

    ContextResponseSchema.parse(response);
    
    span.setAttributes({
      success: true,
      latency_ms: Date.now() - startTime,
      refs_requested: contextRequest.refs.length,
      refs_resolved: contexts.length,
      total_tokens: totalTokens,
    });

    reply.header('content-type', 'application/json; charset=utf-8');
    return JSON.parse(deterministicStringify(response));

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(400);
    throw error;
    
  } finally {
    span.end();
  }
});

// Phase 2: Cross-reference endpoint
fastify.post('/v1/spi/xref', async (request, reply): Promise<XrefResponse> => {
  const span = LensTracer.createChildSpan('spi_xref');
  const startTime = Date.now();

  try {
    const xrefRequest = XrefRequestSchema.parse(request.body);
    
    // Mock cross-reference data
    const mockHit = {
      file: 'src/example.ts',
      line: 42,
      col: 8,
      lang: 'typescript',
      snippet: 'function example() {',
      score: 0.95,
      why: ['symbol'] as ('symbol' | 'exact' | 'fuzzy' | 'struct' | 'semantic' | 'structural' | 'subtoken')[],
      symbol_kind: 'function' as const,
      ref: 'lens://mock-sha/src/example.ts@abc123#L42:42|B1024:1040',
      source_hash: 'abc123',
      byte_start: 1024,
      byte_end: 1040,
    };

    const response: XrefResponse = {
      symbol_id: 'example_func_42',
      symbol_name: 'example',
      definitions: [mockHit],
      references: [
        { ...mockHit, line: 84, why: ['symbol'] as const },
        { ...mockHit, line: 126, why: ['symbol'] as const },
      ],
      total: 3,
    };

    XrefResponseSchema.parse(response);
    
    span.setAttributes({
      success: true,
      latency_ms: Date.now() - startTime,
      definitions_count: response.definitions.length,
      references_count: response.references.length,
    });

    reply.header('content-type', 'application/json; charset=utf-8');
    return JSON.parse(deterministicStringify(response));

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(400);
    throw error;
    
  } finally {
    span.end();
  }
});

// Phase 2: Symbols listing endpoint
fastify.get('/v1/spi/symbols', async (request, reply): Promise<SymbolsListResponse> => {
  const span = LensTracer.createChildSpan('spi_symbols_list');
  const startTime = Date.now();

  try {
    const symbolsRequest = SymbolsListRequestSchema.parse(request.query);
    const page = symbolsRequest.page || 0;
    const pageSize = symbolsRequest.page_size || 100;

    // Mock symbols data
    const symbols = Array.from({ length: pageSize }, (_, i) => ({
      symbol_id: `symbol_${page * pageSize + i}`,
      name: `function_${page * pageSize + i}`,
      kind: 'function' as const,
      file_path: `src/file_${Math.floor((page * pageSize + i) / 10)}.ts`,
      line: (page * pageSize + i) % 100 + 1,
      ref: `lens://${symbolsRequest.repo_sha}/src/file_${Math.floor((page * pageSize + i) / 10)}.ts@hash${i}#L${(page * pageSize + i) % 100 + 1}:${(page * pageSize + i) % 100 + 1}|B${i * 50}:${i * 50 + 20}`,
      lang: 'typescript',
    }));

    const response: SymbolsListResponse = {
      symbols,
      total: 10000, // Mock total
      page,
      page_size: pageSize,
      has_next: page * pageSize + pageSize < 10000,
    };

    SymbolsListResponseSchema.parse(response);
    
    span.setAttributes({
      success: true,
      latency_ms: Date.now() - startTime,
      symbols_returned: symbols.length,
      page,
      page_size: pageSize,
    });

    reply.header('content-type', 'application/json; charset=utf-8');
    return JSON.parse(deterministicStringify(response));

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(400);
    throw error;
    
  } finally {
    span.end();
  }
});

// Phase 3: SSE Streaming Search endpoint
fastify.get('/v1/spi/search:stream', async (request, reply) => {
  const span = LensTracer.createChildSpan('spi_search_stream');
  const startTime = Date.now();
  const traceId = request.headers['x-trace-id'] as string || uuidv4();

  try {
    // Parse query parameters for streaming search
    const spiRequest = SpiSearchRequestSchema.parse(request.query);
    const budgetMs = spiRequest.budget_ms || 2000;

    span.setAttributes({
      'request.repo_sha': spiRequest.repo_sha,
      'request.query': spiRequest.q,
      'request.mode': spiRequest.mode,
      'request.budget_ms': budgetMs,
      'trace_id': traceId,
      'streaming': true,
    });

    // Set up SSE headers
    reply.raw.writeHead(200, {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*',
    });

    // Helper function to write SSE data
    const writeSSE = (data: any) => {
      reply.raw.write(`data: ${JSON.stringify(data)}\n\n`);
    };

    // Perform search using existing engine
    const result = await searchEngine.search({
      trace_id: traceId,
      repo_sha: spiRequest.repo_sha,
      query: spiRequest.q,
      mode: spiRequest.mode,
      k: spiRequest.k,
      fuzzy_distance: spiRequest.fuzzy,
      started_at: new Date(),
      stages: [],
    });

    const totalLatency = Date.now() - startTime;
    const timedOut = totalLatency > budgetMs;

    // Stream hits as JSONL with token budget management
    const tokenBudget = spiRequest.token_budget || 10000;
    let currentTokens = 0;
    let hitsSent = 0;
    let budgetExceeded = false;
    
    for (const hit of result.hits) {
      const sourceHash = calculateSourceHash(hit.snippet || '');
      const byteStart = hit.byte_offset || 0;
      const byteEnd = byteStart + (hit.span_len || hit.snippet?.length || 0);
      
      const spiHit = {
        ...hit,
        source_hash: sourceHash,
        byte_start: byteStart,
        byte_end: byteEnd,
        ref: generateStableRef(
          spiRequest.repo_sha,
          hit.file,
          sourceHash,
          hit.line,
          hit.line + (hit.snippet?.split('\n').length || 1) - 1,
          byteStart,
          byteEnd,
          hit.ast_path
        )
      };

      const hitTokens = estimateObjectTokens(spiHit);
      if (currentTokens + hitTokens > tokenBudget) {
        budgetExceeded = true;
        break;
      }
      
      writeSSE(spiHit);
      currentTokens += hitTokens;
      hitsSent++;
    }

    // Send completion event
    const versionInfo = getVersionInfo();
    writeSSE({
      _type: 'completion',
      total: result.hits.length,
      hits_sent: hitsSent,
      latency_ms: {
        stage_a: result.stage_a_latency || 0,
        stage_b: result.stage_b_latency || 0,
        stage_c: result.stage_c_latency,
        total: totalLatency,
      },
      trace_id: traceId,
      api_version: versionInfo.api_version,
      token_usage: {
        used: currentTokens,
        budget: tokenBudget,
        budget_exceeded: budgetExceeded,
      },
      index_version: versionInfo.index_version,
      policy_version: versionInfo.policy_version,
      timed_out: timedOut,
    });

    span.setAttributes({
      success: true,
      hits_count: result.hits.length,
      total_latency_ms: totalLatency,
      timed_out: timedOut,
      streaming: true,
    });

    reply.raw.end();

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    // Send error as SSE
    reply.raw.write(`data: ${JSON.stringify({
      _type: 'error',
      error: errorMsg,
      trace_id: traceId,
      timestamp: new Date().toISOString(),
    })}\n\n`);
    
    reply.raw.end();
    
  } finally {
    span.end();
  }
});

// Phase 0: SPI Index endpoint (facade over existing indexing)
fastify.post('/v1/spi/index', async (request, reply) => {
  const span = LensTracer.createChildSpan('spi_index');
  const startTime = Date.now();

  try {
    // For now, return a mock success response
    // In a real implementation, this would trigger indexing
    const response = {
      success: true,
      message: 'Index operation queued',
      timestamp: new Date().toISOString(),
      latency_ms: Date.now() - startTime,
    };

    span.setAttributes({
      success: true,
      latency_ms: Date.now() - startTime,
    });

    reply.header('content-type', 'application/json; charset=utf-8');
    return JSON.parse(deterministicStringify(response));

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    throw error;
    
  } finally {
    span.end();
  }
});

// ==== LSP SPI ENDPOINTS ====

// Import LSP service
import { globalLSPService } from '../lsp/service.js';
import {
  LSPCapabilitiesResponseSchema,
  LSPDiagnosticsRequestSchema,
  LSPDiagnosticsResponseSchema,
  LSPFormatRequestSchema,
  LSPFormatResponseSchema,
  LSPSelectionRangesRequestSchema,
  LSPSelectionRangesResponseSchema,
  LSPFoldingRangesRequestSchema,
  LSPFoldingRangesResponseSchema,
  LSPPrepareRenameRequestSchema,
  LSPPrepareRenameResponseSchema,
  type LSPCapabilitiesResponse,
  type LSPDiagnosticsRequest,
  type LSPDiagnosticsResponse,
  type LSPFormatRequest,
  type LSPFormatResponse,
  type LSPSelectionRangesRequest,
  type LSPSelectionRangesResponse,
  type LSPFoldingRangesRequest,
  type LSPFoldingRangesResponse,
  type LSPPrepareRenameRequest,
  type LSPPrepareRenameResponse,
} from '../types/api.js';

// LSP SPI: Capabilities endpoint
fastify.get('/v1/spi/lsp/capabilities', async (request, reply): Promise<LSPCapabilitiesResponse> => {
  const span = LensTracer.createChildSpan('lsp_spi_capabilities');
  const startTime = Date.now();
  const traceId = request.headers['x-trace-id'] as string || uuidv4();

  try {
    const { repo_sha } = request.query as { repo_sha?: string };
    
    const capabilities = await globalLSPService.getCapabilities(repo_sha);
    const duration_ms = Date.now() - startTime;
    
    span.setAttributes({
      success: true,
      duration_ms,
      languages_count: capabilities.languages.length,
      repo_sha: repo_sha || 'all',
    });
    
    // Check SLA (capabilities should be <100ms)
    if (duration_ms > 100) {
      fastify.log.warn(`LSP capabilities SLA breach: ${duration_ms}ms > 100ms`);
    }
    
    return capabilities;
    
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    const duration_ms = Date.now() - startTime;
    
    span.recordException(error);
    span.setAttributes({
      success: false,
      duration_ms,
      error: errorMsg,
    });
    
    fastify.log.error(`LSP capabilities error: ${errorMsg}`);
    
    return reply.status(500).send({
      error: 'LSP capabilities failed',
      message: errorMsg,
      duration_ms,
      trace_id: traceId,
    });
  } finally {
    span.end();
  }
});

// LSP SPI: Diagnostics endpoint
fastify.post('/v1/spi/lsp/diagnostics', async (request, reply): Promise<LSPDiagnosticsResponse> => {
  const span = LensTracer.createChildSpan('lsp_spi_diagnostics');
  const startTime = Date.now();
  const traceId = request.headers['x-trace-id'] as string || uuidv4();

  try {
    const diagnosticsRequest = LSPDiagnosticsRequestSchema.parse(request.body);
    const budget = diagnosticsRequest.budget_ms || 5000;
    
    // Check budget and return 429 if too small
    if (budget < 500) {
      return reply.status(429).send({
        error: 'Budget too small',
        message: 'Minimum budget_ms is 500ms for diagnostics',
        duration_ms: Date.now() - startTime,
        trace_id: traceId,
      });
    }
    
    const response = await globalLSPService.getDiagnostics(diagnosticsRequest);
    
    span.setAttributes({
      success: true,
      duration_ms: response.duration_ms,
      files_count: diagnosticsRequest.files.length,
      total_diagnostics: response.diags.reduce((sum, d) => sum + d.items.length, 0),
      timed_out: response.timed_out || false,
      budget_ms: budget,
    });
    
    // Check SLA (diagnostics should be <3000ms)
    if (response.duration_ms > 3000) {
      fastify.log.warn(`LSP diagnostics SLA breach: ${response.duration_ms}ms > 3000ms`);
    }
    
    // Return 504 if timed out
    if (response.timed_out) {
      return reply.status(504).send({
        ...response,
        error: 'Request timed out',
        trace_id: traceId,
      });
    }
    
    return response;
    
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    const duration_ms = Date.now() - startTime;
    
    span.recordException(error);
    span.setAttributes({
      success: false,
      duration_ms,
      error: errorMsg,
    });
    
    fastify.log.error(`LSP diagnostics error: ${errorMsg}`);
    
    return reply.status(500).send({
      error: 'LSP diagnostics failed',
      message: errorMsg,
      duration_ms,
      trace_id: traceId,
    });
  } finally {
    span.end();
  }
});

// LSP SPI: Format endpoint
fastify.post('/v1/spi/lsp/format', async (request, reply): Promise<LSPFormatResponse> => {
  const span = LensTracer.createChildSpan('lsp_spi_format');
  const startTime = Date.now();
  const traceId = request.headers['x-trace-id'] as string || uuidv4();

  try {
    const formatRequest = LSPFormatRequestSchema.parse(request.body);
    const budget = formatRequest.budget_ms || 3000;
    
    // Check budget
    if (budget < 500) {
      return reply.status(429).send({
        error: 'Budget too small',
        message: 'Minimum budget_ms is 500ms for format',
        duration_ms: Date.now() - startTime,
        trace_id: traceId,
      });
    }
    
    const response = await globalLSPService.format(formatRequest);
    
    span.setAttributes({
      success: true,
      duration_ms: response.duration_ms,
      edits_count: response.edits.length,
      idempotent: response.idempotent,
      budget_ms: budget,
    });
    
    // Check SLA (format should be <2000ms)
    if (response.duration_ms > 2000) {
      fastify.log.warn(`LSP format SLA breach: ${response.duration_ms}ms > 2000ms`);
    }
    
    return response;
    
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    const duration_ms = Date.now() - startTime;
    
    span.recordException(error);
    span.setAttributes({
      success: false,
      duration_ms,
      error: errorMsg,
    });
    
    fastify.log.error(`LSP format error: ${errorMsg}`);
    
    return reply.status(500).send({
      error: 'LSP format failed',
      message: errorMsg,
      duration_ms,
      trace_id: traceId,
    });
  } finally {
    span.end();
  }
});

// LSP SPI: Selection Ranges endpoint
fastify.post('/v1/spi/lsp/selectionRanges', async (request, reply): Promise<LSPSelectionRangesResponse> => {
  const span = LensTracer.createChildSpan('lsp_spi_selection_ranges');
  const startTime = Date.now();
  const traceId = request.headers['x-trace-id'] as string || uuidv4();

  try {
    const selectionRequest = LSPSelectionRangesRequestSchema.parse(request.body);
    const budget = selectionRequest.budget_ms || 2000;
    
    if (budget < 300) {
      return reply.status(429).send({
        error: 'Budget too small',
        message: 'Minimum budget_ms is 300ms for selectionRanges',
        duration_ms: Date.now() - startTime,
        trace_id: traceId,
      });
    }
    
    const response = await globalLSPService.getSelectionRanges(selectionRequest);
    
    span.setAttributes({
      success: true,
      duration_ms: response.duration_ms,
      refs_count: selectionRequest.refs.length,
      total_selections: response.chains.reduce((sum, chain) => sum + chain.length, 0),
      budget_ms: budget,
    });
    
    return response;
    
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    const duration_ms = Date.now() - startTime;
    
    span.recordException(error);
    span.setAttributes({
      success: false,
      duration_ms,
      error: errorMsg,
    });
    
    fastify.log.error(`LSP selection ranges error: ${errorMsg}`);
    
    return reply.status(500).send({
      error: 'LSP selection ranges failed',
      message: errorMsg,
      duration_ms,
      trace_id: traceId,
    });
  } finally {
    span.end();
  }
});

// LSP SPI: Folding Ranges endpoint
fastify.post('/v1/spi/lsp/foldingRanges', async (request, reply): Promise<LSPFoldingRangesResponse> => {
  const span = LensTracer.createChildSpan('lsp_spi_folding_ranges');
  const startTime = Date.now();
  const traceId = request.headers['x-trace-id'] as string || uuidv4();

  try {
    const foldingRequest = LSPFoldingRangesRequestSchema.parse(request.body);
    const budget = foldingRequest.budget_ms || 2000;
    
    if (budget < 300) {
      return reply.status(429).send({
        error: 'Budget too small',
        message: 'Minimum budget_ms is 300ms for foldingRanges',
        duration_ms: Date.now() - startTime,
        trace_id: traceId,
      });
    }
    
    const response = await globalLSPService.getFoldingRanges(foldingRequest);
    
    span.setAttributes({
      success: true,
      duration_ms: response.duration_ms,
      files_count: foldingRequest.files.length,
      total_folds: response.folds.reduce((sum, f) => sum + f.ranges.length, 0),
      budget_ms: budget,
    });
    
    return response;
    
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    const duration_ms = Date.now() - startTime;
    
    span.recordException(error);
    span.setAttributes({
      success: false,
      duration_ms,
      error: errorMsg,
    });
    
    fastify.log.error(`LSP folding ranges error: ${errorMsg}`);
    
    return reply.status(500).send({
      error: 'LSP folding ranges failed',
      message: errorMsg,
      duration_ms,
      trace_id: traceId,
    });
  } finally {
    span.end();
  }
});

// LSP SPI: Prepare Rename endpoint
fastify.post('/v1/spi/lsp/prepareRename', async (request, reply): Promise<LSPPrepareRenameResponse> => {
  const span = LensTracer.createChildSpan('lsp_spi_prepare_rename');
  const startTime = Date.now();
  const traceId = request.headers['x-trace-id'] as string || uuidv4();

  try {
    const prepareRenameRequest = LSPPrepareRenameRequestSchema.parse(request.body);
    const budget = prepareRenameRequest.budget_ms || 3000;
    
    if (budget < 500) {
      return reply.status(429).send({
        error: 'Budget too small',
        message: 'Minimum budget_ms is 500ms for prepareRename',
        duration_ms: Date.now() - startTime,
        trace_id: traceId,
      });
    }
    
    const response = await globalLSPService.prepareRename(prepareRenameRequest);
    
    span.setAttributes({
      success: true,
      duration_ms: response.duration_ms,
      allowed: response.allowed,
      budget_ms: budget,
    });
    
    return response;
    
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    const duration_ms = Date.now() - startTime;
    
    span.recordException(error);
    span.setAttributes({
      success: false,
      duration_ms,
      error: errorMsg,
    });
    
    fastify.log.error(`LSP prepare rename error: ${errorMsg}`);
    
    return reply.status(500).send({
      allowed: false,
      reason: `Error: ${errorMsg}`,
      duration_ms,
    });
  } finally {
    span.end();
  }
});

// LSP SPI: Rename endpoint - Sprint B
fastify.post('/v1/spi/lsp/rename', async (request, reply): Promise<LSPRenameResponse> => {
  const span = LensTracer.createChildSpan('lsp_spi_rename');
  const startTime = Date.now();
  const traceId = request.headers['x-trace-id'] as string || uuidv4();

  try {
    const renameRequest = LSPRenameRequestSchema.parse(request.body);
    const budget = renameRequest.budget_ms || 5000;
    
    if (budget < 1000) {
      return reply.status(429).send({
        error: 'Budget too small',
        message: 'Minimum budget_ms is 1000ms for rename operations',
        duration_ms: Date.now() - startTime,
        trace_id: traceId,
      });
    }
    
    const response = await globalLSPService.rename(renameRequest);
    
    span.setAttributes({
      success: true,
      duration_ms: response.duration_ms,
      edit_count: response.workspaceEdit?.changes ? Object.keys(response.workspaceEdit.changes).length : 0,
      budget_ms: budget,
    });
    
    return response;
    
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    const duration_ms = Date.now() - startTime;
    
    span.recordException(error);
    span.setAttributes({
      success: false,
      duration_ms,
      error: errorMsg,
    });
    
    fastify.log.error(`LSP rename error: ${errorMsg}`);
    
    return reply.status(500).send({
      workspace_edit: null,
      duration_ms,
      trace_id: traceId,
    });
  } finally {
    span.end();
  }
});

// LSP SPI: Code Actions endpoint - Sprint B
fastify.post('/v1/spi/lsp/codeActions', async (request, reply): Promise<LSPCodeActionsResponse> => {
  const span = LensTracer.createChildSpan('lsp_spi_code_actions');
  const startTime = Date.now();
  const traceId = request.headers['x-trace-id'] as string || uuidv4();

  try {
    const codeActionsRequest = LSPCodeActionsRequestSchema.parse(request.body);
    const budget = codeActionsRequest.budget_ms || 2000;
    
    if (budget < 300) {
      return reply.status(429).send({
        error: 'Budget too small',
        message: 'Minimum budget_ms is 300ms for code actions',
        duration_ms: Date.now() - startTime,
        trace_id: traceId,
      });
    }
    
    const response = await globalLSPService.getCodeActions(codeActionsRequest);
    
    span.setAttributes({
      success: true,
      duration_ms: response.duration_ms,
      action_count: response.actions.length,
      budget_ms: budget,
    });
    
    return response;
    
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    const duration_ms = Date.now() - startTime;
    
    span.recordException(error);
    span.setAttributes({
      success: false,
      duration_ms,
      error: errorMsg,
    });
    
    fastify.log.error(`LSP code actions error: ${errorMsg}`);
    
    return reply.status(500).send({
      actions: [],
      duration_ms,
      trace_id: traceId,
    });
  } finally {
    span.end();
  }
});

// LSP SPI: Hierarchy endpoint - Sprint B
fastify.post('/v1/spi/lsp/hierarchy', async (request, reply): Promise<LSPHierarchyResponse> => {
  const span = LensTracer.createChildSpan('lsp_spi_hierarchy');
  const startTime = Date.now();
  const traceId = request.headers['x-trace-id'] as string || uuidv4();

  try {
    const hierarchyRequest = LSPHierarchyRequestSchema.parse(request.body);
    const budget = hierarchyRequest.budget_ms || 3000;
    
    if (budget < 500) {
      return reply.status(429).send({
        error: 'Budget too small',
        message: 'Minimum budget_ms is 500ms for hierarchy operations',
        duration_ms: Date.now() - startTime,
        trace_id: traceId,
      });
    }
    
    const response = await globalLSPService.getHierarchy(hierarchyRequest);
    
    span.setAttributes({
      success: true,
      duration_ms: response.duration_ms,
      node_count: response.nodes.length,
      hierarchy_kind: hierarchyRequest.kind,
      budget_ms: budget,
    });
    
    return response;
    
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    const duration_ms = Date.now() - startTime;
    
    span.recordException(error);
    span.setAttributes({
      success: false,
      duration_ms,
      error: errorMsg,
    });
    
    fastify.log.error(`LSP hierarchy error: ${errorMsg}`);
    
    return reply.status(500).send({
      nodes: [],
      duration_ms,
      trace_id: traceId,
    });
  } finally {
    span.end();
  }
});

// Get precision optimization status
fastify.get('/policy/precision/status', async (request, reply) => {
  const span = LensTracer.createChildSpan('api_precision_status');
  const startTime = Date.now();

  try {
    const status = searchEngine.getPrecisionOptimizationStatus();
    const latency = Date.now() - startTime;

    span.setAttributes({
      success: true,
      latency_ms: latency,
      block_a_enabled: status.block_a_enabled,
      block_b_enabled: status.block_b_enabled,
      block_c_enabled: status.block_c_enabled
    });

    return {
      success: true,
      message: 'Precision optimization status retrieved',
      status,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Unknown error';
    span.recordException(error as Error);
    span.setAttributes({ success: false, error: errorMsg });
    
    reply.status(500);
    return {
      success: false,
      error: 'Failed to get precision optimization status',
      message: errorMsg
    };
  } finally {
    span.end();
  }
});

// Start server with dynamic port allocation
export async function startServer(port?: number, host: string = '0.0.0.0') {
  try {
    const { portManager, getServerPort, getMetricsPort } = await import('../config/ports.js');
    
    // Use provided port or get dynamically allocated port
    const serverPort = port || await getServerPort();
    const metricsPort = await getMetricsPort();
    
    // Initialize search engine
    await searchEngine.initialize();
    
    // Initialize RAPTOR metrics telemetry
    const metricsTelemetry = new MetricsTelemetry({
      collection_interval_ms: 60000, // 1 minute
      retention_days: 7
    });
    
    // Initialize RAPTOR configuration and rollout manager
    const configManager = new ConfigRolloutManager({
      enabled: process.env.RAPTOR_ENABLED === 'true',
      rollout_percentage: parseInt(process.env.RAPTOR_ROLLOUT_PERCENTAGE || '0'),
      rollout: {
        strategy: 'percentage',
        target_percentage: parseInt(process.env.RAPTOR_TARGET_PERCENTAGE || '0'),
        ramp_up_duration_hours: parseInt(process.env.RAPTOR_RAMP_DURATION_HOURS || '24'),
        canary_percentage: 1,
        monitoring_window_minutes: 30
      }
    });
    
    // Register all endpoint types
    await registerBenchmarkEndpoints(fastify);
    await registerPrecisionMonitoringEndpoints(fastify);
    await registerRaptorMetricsEndpoints(fastify, metricsTelemetry);
    await registerConfigEndpoints(fastify, configManager);
    
    await fastify.listen({ port: serverPort, host });
    
    // Extend reservation to keep ports reserved while server is running
    await portManager.extendReservation();
    
    console.log(`🚀 Lens server running on http://${host}:${serverPort}`);
    console.log(`📊 Metrics available at http://${host}:${metricsPort}/metrics`);
    console.log(`🔍 Health check at http://${host}:${serverPort}/health`);
    console.log(`🧪 Benchmark endpoints at http://${host}:${serverPort}/bench/`);
    console.log(`🔧 SPI endpoints at http://${host}:${serverPort}/v1/spi/`);
    console.log(`   • POST /v1/spi/search - Enhanced search with stable refs`);
    console.log(`   • GET  /v1/spi/health - Extended health with SLA data`);
    console.log(`   • GET  /v1/spi/resolve - Resolve lens:// references`);
    console.log(`   • POST /v1/spi/context - Batch context resolution`);
    console.log(`   • POST /v1/spi/xref - Cross-reference lookup`);
    console.log(`   • GET  /v1/spi/symbols - Symbol listing with pagination`);
    console.log(`   • GET  /v1/spi/search:stream - SSE streaming search`);
    console.log(`   • POST /v1/spi/index - Index management`);
    console.log(`🧠 LSP SPI endpoints at http://${host}:${serverPort}/v1/spi/lsp/`);
    console.log(`   • GET  /v1/spi/lsp/capabilities - Language capabilities discovery`);
    console.log(`   • POST /v1/spi/lsp/diagnostics - Fast verify gate with error/warning detection`);
    console.log(`   • POST /v1/spi/lsp/format - Idempotent code formatting`);
    console.log(`   • POST /v1/spi/lsp/selectionRanges - Snippet fences for selection`);
    console.log(`   • POST /v1/spi/lsp/foldingRanges - Code folding ranges`);
    console.log(`   • POST /v1/spi/lsp/prepareRename - Rename validation`);
    console.log(`   • POST /v1/spi/lsp/rename - Multi-file workspace edits for refactoring`);
    console.log(`   • POST /v1/spi/lsp/codeActions - Quick fixes and refactoring suggestions`);
    console.log(`   • POST /v1/spi/lsp/hierarchy - Call/type hierarchy for impact analysis`);
    console.log(`📋 Port configuration written to .port-config.json`);
    
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
      await globalLSPService.shutdown();
      await fastify.close();
      
      // Release port reservations on shutdown
      const { portManager } = await import('../config/ports.js');
      await portManager.releaseReservations();
      
      console.log('Server shut down successfully');
      process.exit(0);
    } catch (err) {
      console.error('Error during shutdown:', err);
      process.exit(1);
    }
  });
});

export { fastify, searchEngine };