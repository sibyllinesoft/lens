/**
 * Benchmark API Endpoints
 * Exposes benchmarking functionality per TODO.md specifications
 */

import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { z } from 'zod';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import { createBenchmarkOrchestrator, type BenchmarkOrchestrationConfig } from '../benchmark/index.js';
import { BenchmarkConfigSchema } from '../types/benchmark.js';

// API request schemas
const RunBenchmarkRequestSchema = z.object({
  suite: z.array(z.enum(['codesearch', 'structural', 'docs'])).default(['codesearch', 'structural']),
  systems: z.array(z.string()).default(['lex', '+symbols', '+symbols+semantic']),
  slices: z.union([
    z.literal('SMOKE_DEFAULT'),
    z.literal('ALL'),
    z.array(z.string())
  ]).default('SMOKE_DEFAULT'),
  seeds: z.number().int().min(1).max(5).default(1),
  cache_mode: z.union([
    z.enum(['warm', 'cold']),
    z.array(z.enum(['warm', 'cold']))
  ]).default('warm'),
  robustness: z.boolean().default(false),
  metamorphic: z.boolean().default(false),
  trace_id: z.string().uuid().optional()
});

const HealthCheckRequestSchema = z.object({
  include_details: z.boolean().default(false)
});

type RunBenchmarkRequest = z.infer<typeof RunBenchmarkRequestSchema>;
type HealthCheckRequest = z.infer<typeof HealthCheckRequestSchema>;

export async function registerBenchmarkEndpoints(
  fastify: FastifyInstance,
  options: {
    workingDir?: string;
    outputDir?: string;
    natsUrl?: string;
    repositories?: Array<{ name: string; path: string }>;
  } = {}
) {
  
  // Default configuration
  const defaultConfig: BenchmarkOrchestrationConfig = {
    workingDir: options.workingDir || process.cwd(),
    outputDir: options.outputDir || path.join(process.cwd(), 'benchmark-results'),
    natsUrl: options.natsUrl || 'nats://localhost:4222',
    repositories: options.repositories || [
      { name: 'lens', path: process.cwd() } // Self-benchmark by default
    ]
  };

  /**
   * POST /bench/run - Run benchmark suite
   */
  fastify.post<{ Body: RunBenchmarkRequest }>(
    '/bench/run',
    {
      schema: {
        body: {
          type: 'object',
          properties: {
            suite: { type: 'array', items: { type: 'string' } },
            systems: { type: 'array', items: { type: 'string' } },
            slices: { 
              oneOf: [
                { type: 'string', enum: ['SMOKE_DEFAULT', 'ALL'] },
                { type: 'array', items: { type: 'string' } }
              ]
            },
            seeds: { type: 'integer', minimum: 1, maximum: 5 },
            cache_mode: {
              oneOf: [
                { type: 'string', enum: ['warm', 'cold'] },
                { type: 'array', items: { type: 'string', enum: ['warm', 'cold'] } }
              ]
            },
            robustness: { type: 'boolean' },
            metamorphic: { type: 'boolean' },
            trace_id: { type: 'string', format: 'uuid' }
          }
        },
        response: {
          200: {
            type: 'object',
            properties: {
              trace_id: { type: 'string' },
              status: { type: 'string' },
              benchmark_run: { type: 'object' },
              reports: {
                type: 'object',
                properties: {
                  pdf_path: { type: 'string' },
                  markdown_path: { type: 'string' },
                  json_path: { type: 'string' }
                }
              },
              artifacts: { type: 'array', items: { type: 'string' } },
              duration_ms: { type: 'number' }
            }
          }
        }
      }
    },
    async (request: FastifyRequest<{ Body: RunBenchmarkRequest }>, reply: FastifyReply) => {
      const startTime = Date.now();
      
      try {
        // Validate request
        const validatedRequest = RunBenchmarkRequestSchema.parse(request.body);
        
        const traceId = validatedRequest.trace_id || uuidv4();
        
        fastify.log.info({ trace_id: traceId }, 'Starting benchmark suite');
        
        // Create orchestrator
        const orchestrator = createBenchmarkOrchestrator(defaultConfig);
        
        // Determine suite type
        const suiteType = validatedRequest.robustness || validatedRequest.metamorphic ? 'full' : 'smoke';
        
        // Execute benchmark
        const result = await orchestrator.runCompleteBenchmark(
          {
            ...validatedRequest,
            trace_id: traceId
          },
          suiteType
        );
        
        // Cleanup
        await orchestrator.cleanup();
        
        const duration = Date.now() - startTime;
        
        fastify.log.info({ 
          trace_id: traceId, 
          duration_ms: duration,
          suite_type: suiteType
        }, 'Benchmark suite completed');
        
        reply.code(200).send({
          trace_id: traceId,
          status: 'completed',
          benchmark_run: result.benchmark_run,
          reports: result.reports,
          artifacts: result.artifacts,
          duration_ms: duration
        });
        
      } catch (error) {
        fastify.log.error(error, 'Benchmark execution failed');
        
        reply.code(500).send({
          error: 'benchmark_execution_failed',
          message: error instanceof Error ? error.message : String(error),
          trace_id: request.body.trace_id || 'unknown'
        });
      }
    }
  );

  /**
   * POST /bench/smoke - Quick smoke test (optimized endpoint)
   */
  fastify.post('/bench/smoke', async (request, reply) => {
    const traceId = uuidv4();
    const startTime = Date.now();
    
    try {
      fastify.log.info({ trace_id: traceId }, 'Starting smoke test');
      
      const orchestrator = createBenchmarkOrchestrator(defaultConfig);
      
      const result = await orchestrator.runSmoke({
        trace_id: traceId,
        suite: ['codesearch'],
        systems: ['lex', '+symbols'],
        cache_mode: 'warm'
      });
      
      await orchestrator.cleanup();
      
      reply.code(200).send({
        trace_id: traceId,
        status: 'completed',
        duration_ms: Date.now() - startTime,
        promotion_gate: result.promotion_gate || 'unknown',
        reports: result.reports
      });
      
    } catch (error) {
      fastify.log.error(error, 'Smoke test failed');
      reply.code(500).send({
        error: 'smoke_test_failed',
        message: error instanceof Error ? error.message : String(error),
        trace_id: traceId
      });
    }
  });

  /**
   * GET /bench/health - Benchmark system health check
   */
  fastify.get<{ Querystring: HealthCheckRequest }>(
    '/bench/health',
    {
      schema: {
        querystring: {
          type: 'object',
          properties: {
            include_details: { type: 'boolean' }
          }
        },
        response: {
          200: {
            type: 'object',
            properties: {
              status: { type: 'string', enum: ['healthy', 'degraded', 'unhealthy'] },
              components: { type: 'object' },
              details: { type: 'object' }
            }
          }
        }
      }
    },
    async (request, reply) => {
      try {
        const { include_details } = HealthCheckRequestSchema.parse(request.query);
        
        const orchestrator = createBenchmarkOrchestrator(defaultConfig);
        const health = await orchestrator.healthCheck();
        await orchestrator.cleanup();
        
        const response: any = {
          status: health.status,
          components: health.components
        };
        
        if (include_details) {
          response.details = health.details;
        }
        
        reply.code(200).send(response);
        
      } catch (error) {
        fastify.log.error(error, 'Health check failed');
        reply.code(500).send({
          status: 'unhealthy',
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }
  );

  /**
   * GET /bench/config - Get current benchmark configuration
   */
  fastify.get('/bench/config', async (request, reply) => {
    reply.code(200).send({
      config: {
        working_dir: defaultConfig.workingDir,
        output_dir: defaultConfig.outputDir,
        nats_url: defaultConfig.natsUrl,
        repositories: defaultConfig.repositories.map(r => r.name)
      },
      defaults: {
        systems: ['lex', '+symbols', '+symbols+semantic'],
        suite: ['codesearch', 'structural'],
        seeds: 1,
        cache_mode: 'warm'
      }
    });
  });

  /**
   * GET /bench/status/:trace_id - Get benchmark run status (placeholder)
   */
  fastify.get<{ Params: { trace_id: string } }>(
    '/bench/status/:trace_id',
    async (request, reply) => {
      const { trace_id } = request.params;
      
      // In a real implementation, this would query the NATS telemetry system
      reply.code(200).send({
        trace_id,
        status: 'unknown',
        message: 'Status tracking not implemented - check NATS telemetry topics'
      });
    }
  );

  fastify.log.info('Benchmark endpoints registered');
}