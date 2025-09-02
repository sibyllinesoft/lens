"use strict";
/**
 * OpenTelemetry tracer setup for Lens
 * Full distributed tracing with spans, metrics, and traces
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.meter = exports.tracer = exports.LensTracer = void 0;
const sdk_node_1 = require("@opentelemetry/sdk-node");
const auto_instrumentations_node_1 = require("@opentelemetry/auto-instrumentations-node");
const exporter_jaeger_1 = require("@opentelemetry/exporter-jaeger");
const exporter_prometheus_1 = require("@opentelemetry/exporter-prometheus");
const api_1 = require("@opentelemetry/api");
const uuid_1 = require("uuid");
// Initialize OpenTelemetry SDK
const sdk = new sdk_node_1.NodeSDK({
    traceExporter: new exporter_jaeger_1.JaegerExporter({
        endpoint: process.env.JAEGER_ENDPOINT || 'http://localhost:14268/api/traces',
    }),
    metricReader: new exporter_prometheus_1.PrometheusExporter({
        port: 9464, // Prometheus metrics port
    }, () => {
        console.log('Prometheus metrics server started on port 9464');
    }),
    instrumentations: [(0, auto_instrumentations_node_1.getNodeAutoInstrumentations)()],
});
// Start the SDK
sdk.start();
console.log('OpenTelemetry tracing initialized');
// Get tracer and meter instances
const tracer = api_1.trace.getTracer('lens-search', '0.1.0');
exports.tracer = tracer;
const meter = api_1.metrics.getMeter('lens-search', '0.1.0');
exports.meter = meter;
// Create metrics
const queryCounter = meter.createCounter('lens_queries_total', {
    description: 'Total number of search queries',
});
const queryDuration = meter.createHistogram('lens_query_duration_ms', {
    description: 'Query duration in milliseconds',
    boundaries: [1, 5, 10, 20, 50, 100, 200, 500, 1000],
});
const stageLatencies = {
    stage_a: meter.createHistogram('lens_stage_a_duration_ms', {
        description: 'Stage A (lexical+fuzzy) duration in milliseconds',
        boundaries: [1, 2, 5, 8, 10, 15, 20],
    }),
    stage_b: meter.createHistogram('lens_stage_b_duration_ms', {
        description: 'Stage B (symbol/AST) duration in milliseconds',
        boundaries: [1, 3, 5, 7, 10, 15, 20],
    }),
    stage_c: meter.createHistogram('lens_stage_c_duration_ms', {
        description: 'Stage C (semantic rerank) duration in milliseconds',
        boundaries: [1, 5, 10, 12, 15, 20, 30],
    }),
};
const candidatesCounter = meter.createHistogram('lens_candidates_count', {
    description: 'Number of candidates processed per stage',
    boundaries: [1, 10, 50, 100, 200, 500, 1000],
});
class LensTracer {
    /**
     * Create a new search context with tracing
     */
    static createSearchContext(query, mode) {
        return {
            trace_id: (0, uuid_1.v4)(),
            query,
            mode: mode,
            k: 50, // Default k value
            fuzzy_distance: 2, // Default fuzzy distance
            started_at: new Date(),
            stages: [],
        };
    }
    /**
     * Start a search span
     */
    static startSearchSpan(ctx) {
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
    static startStageSpan(ctx, stage, method, candidatesIn) {
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
    static endStageSpan(span, ctx, stage, method, candidatesIn, candidatesOut, latencyMs, error) {
        // Update span attributes
        span.setAttributes({
            'lens.candidates_out': candidatesOut,
            'lens.latency_ms': latencyMs,
        });
        if (error) {
            span.setStatus({
                code: api_1.SpanStatusCode.ERROR,
                message: error,
            });
        }
        else {
            span.setStatus({ code: api_1.SpanStatusCode.OK });
        }
        span.end();
        // Record stage result
        const stageResult = {
            stage,
            latency_ms: latencyMs,
            candidates_in: candidatesIn,
            candidates_out: candidatesOut,
            method,
            error,
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
    static endSearchSpan(span, ctx, totalResults, error) {
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
                code: api_1.SpanStatusCode.ERROR,
                message: error,
            });
        }
        else {
            span.setStatus({ code: api_1.SpanStatusCode.OK });
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
    static createChildSpan(name, attributes = {}) {
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
        return api_1.context.active();
    }
    /**
     * Run with specific context
     */
    static withContext(ctx, fn) {
        return api_1.context.with(ctx, fn);
    }
}
exports.LensTracer = LensTracer;
// Graceful shutdown
process.on('SIGTERM', () => {
    sdk.shutdown()
        .then(() => console.log('OpenTelemetry terminated'))
        .catch((error) => console.log('Error terminating OpenTelemetry', error))
        .finally(() => process.exit(0));
});
