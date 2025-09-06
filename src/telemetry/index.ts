// Re-export OpenTelemetry APIs for consistent usage across the codebase
import * as api from '@opentelemetry/api';

export const opentelemetry = {
  trace: api.trace,
  metrics: api.metrics,
  SpanStatusCode: api.SpanStatusCode,
  context: api.context
};

// Re-export tracer components
export { LensTracer, tracer, meter } from './tracer.js';