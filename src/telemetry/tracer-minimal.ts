/**
 * Minimal telemetry tracer for Lens Daemon
 * Provides basic logging without complex OpenTelemetry dependencies
 */

// Simple console-based tracing for daemon mode
console.log('📊 Minimal telemetry initialized for Lens daemon');

export class LensTracer {
  static createSearchContext(query: string, mode: string, repo_sha: string = 'unknown') {
    return {
      trace_id: `trace-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      repo_sha,
      query,
      mode,
      k: 50,
      fuzzy_distance: 2,
      started_at: new Date(),
      stages: [],
    };
  }

  static startSearchSpan(ctx: any) {
    console.log(`🔍 Search started: ${ctx.query} (${ctx.mode})`);
    return { spanId: `span-${Date.now()}` };
  }

  static startStageSpan(ctx: any, stage: string, method: string, candidatesIn: number) {
    console.log(`⚡ Stage ${stage} started: ${method} (${candidatesIn} candidates)`);
    return { spanId: `stage-${stage}-${Date.now()}` };
  }

  static endStageSpan(span: any, ctx: any, stage: string, method: string, candidatesIn: number, candidatesOut: number, latencyMs: number, error?: string) {
    if (error) {
      console.log(`❌ Stage ${stage} failed: ${error} (${latencyMs}ms)`);
    } else {
      console.log(`✅ Stage ${stage} completed: ${candidatesIn} → ${candidatesOut} candidates (${latencyMs}ms)`);
    }
  }

  static endSearchSpan(span: any, ctx: any, totalResults: number, error?: string) {
    const totalLatency = Date.now() - ctx.started_at.getTime();
    if (error) {
      console.log(`❌ Search failed: ${error} (${totalLatency}ms)`);
    } else {
      console.log(`✅ Search completed: ${totalResults} results (${totalLatency}ms)`);
    }
  }

  static createChildSpan(name: string, attributes: Record<string, any> = {}) {
    return { spanId: `child-${name}-${Date.now()}`, attributes };
  }

  static getActiveContext() {
    return {};
  }

  static withContext<T>(ctx: any, fn: () => T): T {
    return fn();
  }
}

// Export minimal tracer
export const tracer = {
  startSpan: (name: string, options?: any) => ({ spanId: `${name}-${Date.now()}` })
};

export const meter = {
  createCounter: (name: string, options?: any) => ({
    add: (value: number, labels?: any) => {
      console.log(`📊 Counter ${name}: +${value}`, labels || '');
    }
  }),
  createHistogram: (name: string, options?: any) => ({
    record: (value: number, labels?: any) => {
      console.log(`📈 Histogram ${name}: ${value}`, labels || '');
    }
  })
};