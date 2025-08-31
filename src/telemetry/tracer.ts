/**
 * OpenTelemetry tracer setup for Lens
 * Full distributed tracing with spans, metrics, and traces
 */

import { NodeSDK } from '@opentelemetry/sdk-node';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { JaegerExporter } from '@opentelemetry/exporter-jaeger';
import { PrometheusExporter } from '@opentelemetry/exporter-prometheus';
import { trace, metrics, SpanStatusCode, context } from '@opentelemetry/api';
import { v4 as uuidv4 } from 'uuid';
import type { SearchContext, StageResult } from '../types/core.js';

// Initialize OpenTelemetry SDK
const sdk = new NodeSDK({
  traceExporter: new JaegerExporter({
    endpoint: process.env['JAEGER_ENDPOINT'] || 'http://localhost:14268/api/traces',
  }),
  metricReader: new PrometheusExporter({
    port: 9464, // Prometheus metrics port
  }, () => {
    console.log('Prometheus metrics server started on port 9464');
  }),
  instrumentations: [getNodeAutoInstrumentations()],
});

// Start the SDK
sdk.start();
console.log('OpenTelemetry tracing initialized');

// Get tracer and meter instances
const tracer = trace.getTracer('lens-search', '0.1.0');
const meter = metrics.getMeter('lens-search', '0.1.0');

// Create metrics
const queryCounter = meter.createCounter('lens_queries_total', {
  description: 'Total number of search queries',
});

const queryDuration = meter.createHistogram('lens_query_duration_ms', {
  description: 'Query duration in milliseconds',
});

const stageLatencies = {
  stage_a: meter.createHistogram('lens_stage_a_duration_ms', {
    description: 'Stage A (lexical+fuzzy) duration in milliseconds',
  }),
  stage_b: meter.createHistogram('lens_stage_b_duration_ms', {
    description: 'Stage B (symbol/AST) duration in milliseconds',
  }),
  stage_c: meter.createHistogram('lens_stage_c_duration_ms', {
    description: 'Stage C (semantic rerank) duration in milliseconds',
  }),
};

const candidatesCounter = meter.createHistogram('lens_candidates_count', {
  description: 'Number of candidates processed per stage',
});

export class LensTracer {
  /**
   * Create a new search context with tracing
   */
  static createSearchContext(query: string, mode: string): SearchContext {
    return {
      trace_id: uuidv4(),
      query,
      mode: mode as any,
      k: 50, // Default k value
      fuzzy_distance: 2, // Default fuzzy distance
      started_at: new Date(),
      stages: [],
    };
  }

  /**
   * Start a search span
   */
  static startSearchSpan(ctx: SearchContext) {
    const span = tracer.startSpan('lens_search', {
      attributes: {
        'lens.query': ctx.query,
        'lens.mode': ctx.mode,
        'lens.trace_id': ctx.trace_id,
        'lens.k': ctx.k,
        'lens.fuzzy_distance': ctx.fuzzy_distance,
      },
    });

    // Record query metrics
    queryCounter.add(1, {
      mode: ctx.mode,
    });

    return span;
  }

  /**
   * Start a processing stage span
   */
  static startStageSpan(
    ctx: SearchContext, 
    stage: 'stage_a' | 'stage_b' | 'stage_c',
    method: string,
    candidatesIn: number
  ) {
    const span = tracer.startSpan(`lens_${stage}`, {
      attributes: {
        'lens.trace_id': ctx.trace_id,
        'lens.stage': stage,
        'lens.method': method,
        'lens.candidates_in': candidatesIn,
      },
    });

    return span;
  }

  /**
   * End a processing stage span with results
   */
  static endStageSpan(
    span: any,
    ctx: SearchContext,
    stage: 'stage_a' | 'stage_b' | 'stage_c',
    method: string,
    candidatesIn: number,
    candidatesOut: number,
    latencyMs: number,
    error?: string
  ) {
    // Update span attributes
    span.setAttributes({
      'lens.candidates_out': candidatesOut,
      'lens.latency_ms': latencyMs,
    });

    if (error) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: error,
      });
    } else {
      span.setStatus({ code: SpanStatusCode.OK });
    }

    span.end();

    // Record stage result
    const stageResult: StageResult = {
      stage,
      latency_ms: latencyMs,
      candidates_in: candidatesIn,
      candidates_out: candidatesOut,
      method,
      ...(error && { error }),
    };
    ctx.stages.push(stageResult);

    // Record metrics
    stageLatencies[stage].record(latencyMs, {
      method,
      success: error ? 'false' : 'true',
    });

    candidatesCounter.record(candidatesIn, {
      stage: `${stage}_in`,
    });

    candidatesCounter.record(candidatesOut, {
      stage: `${stage}_out`,
    });
  }

  /**
   * End search span with final results
   */
  static endSearchSpan(span: any, ctx: SearchContext, totalResults: number, error?: string) {
    const totalLatency = Date.now() - ctx.started_at.getTime();

    span.setAttributes({
      'lens.total_results': totalResults,
      'lens.total_latency_ms': totalLatency,
      'lens.stages_count': ctx.stages.length,
    });

    // Add stage-specific attributes
    ctx.stages.forEach((stage, index) => {
      span.setAttributes({
        [`lens.${stage.stage}.latency_ms`]: stage.latency_ms,
        [`lens.${stage.stage}.candidates_out`]: stage.candidates_out,
        [`lens.${stage.stage}.method`]: stage.method,
      });
    });

    if (error) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: error,
      });
    } else {
      span.setStatus({ code: SpanStatusCode.OK });
    }

    span.end();

    // Record overall query duration
    queryDuration.record(totalLatency, {
      mode: ctx.mode,
      success: error ? 'false' : 'true',
    });
  }

  /**
   * Create a child span for detailed operations
   */
  static createChildSpan(name: string, attributes: Record<string, any> = {}) {
    return tracer.startSpan(name, {
      attributes: {
        ...attributes,
        'component': 'lens-search',
      },
    });
  }

  /**
   * Get current trace context
   */
  static getActiveContext() {
    return context.active();
  }

  /**
   * Run with specific context
   */
  static withContext<T>(ctx: any, fn: () => T): T {
    return context.with(ctx, fn);
  }
}

// Export tracer for direct use
export { tracer, meter };

// Graceful shutdown
process.on('SIGTERM', () => {
  sdk.shutdown()
    .then(() => console.log('OpenTelemetry terminated'))
    .catch((error) => console.log('Error terminating OpenTelemetry', error))
    .finally(() => process.exit(0));
});