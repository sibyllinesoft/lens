/**
 * NATS Telemetry Integration for Benchmark Runs
 * Publishes to topics: lens.bench.plan, lens.bench.run, lens.bench.result
 */

import { connect, NatsConnection, JSONCodec, StringCodec } from 'nats';
import type {
  BenchmarkPlanMessage,
  BenchmarkRunMessage, 
  BenchmarkResultMessage
} from '../types/benchmark.js';

export class NATSTelemetry {
  private connection: NatsConnection | null = null;
  private jsonCodec = JSONCodec();
  private stringCodec = StringCodec();
  
  private readonly topics = {
    PLAN: 'lens.bench.plan',
    RUN: 'lens.bench.run', 
    RESULT: 'lens.bench.result'
  } as const;

  constructor(
    private readonly natsUrl: string = 'nats://localhost:4222',
    private readonly connectionTimeoutMs: number = 5000
  ) {}

  /**
   * Connect to NATS server
   */
  async connect(): Promise<void> {
    try {
      this.connection = await connect({
        servers: this.natsUrl,
        timeout: this.connectionTimeoutMs,
        reconnect: true,
        maxReconnectAttempts: 5,
        reconnectTimeWait: 2000
        // Note: maxPayload option removed as it's not available in current NATS.js version
      });
      
      console.log(`📡 Connected to NATS at ${this.natsUrl}`);
      
      // Handle connection events
      this.connection.closed().then((err) => {
        if (err) {
          console.error('NATS connection closed with error:', err);
        } else {
          console.log('NATS connection closed gracefully');
        }
      });
      
    } catch (error) {
      console.error('Failed to connect to NATS:', error);
      throw error;
    }
  }

  /**
   * Publish benchmark plan message
   */
  async publishPlan(planMessage: BenchmarkPlanMessage): Promise<void> {
    await this.ensureConnected();
    
    try {
      const encoded = this.jsonCodec.encode(planMessage);
      await this.connection!.publish(this.topics.PLAN, encoded);
      
      console.log(`📋 Published plan - Trace: ${planMessage.trace_id}, Queries: ${planMessage.total_queries}`);
      
    } catch (error) {
      console.error('Failed to publish plan message:', error);
      throw error;
    }
  }

  /**
   * Publish benchmark run status message
   */
  async publishRun(runMessage: BenchmarkRunMessage): Promise<void> {
    await this.ensureConnected();
    
    try {
      const encoded = this.jsonCodec.encode(runMessage);
      await this.connection!.publish(this.topics.RUN, encoded);
      
      // Log only significant status updates to avoid spam
      if (runMessage.status === 'started' || runMessage.status === 'failed' || 
          (runMessage.status === 'query_completed' && Math.random() < 0.1)) { // Sample 10% of query completions
        console.log(`🏃 Run update - Trace: ${runMessage.trace_id}, Status: ${runMessage.status}`);
      }
      
    } catch (error) {
      console.error('Failed to publish run message:', error);
      // Don't throw here to avoid breaking benchmark execution
    }
  }

  /**
   * Publish benchmark result message
   */
  async publishResult(resultMessage: BenchmarkResultMessage): Promise<void> {
    await this.ensureConnected();
    
    try {
      const encoded = this.jsonCodec.encode(resultMessage);
      await this.connection!.publish(this.topics.RESULT, encoded);
      
      const gateStatus = resultMessage.promotion_gate_result.passed ? '✅ PASS' : '❌ FAIL';
      console.log(`📊 Published result - Trace: ${resultMessage.trace_id}, Gate: ${gateStatus}, Duration: ${resultMessage.duration_ms}ms`);
      
    } catch (error) {
      console.error('Failed to publish result message:', error);
      throw error;
    }
  }

  /**
   * Subscribe to benchmark plan messages for monitoring
   */
  async subscribePlans(callback: (plan: BenchmarkPlanMessage) => void): Promise<void> {
    await this.ensureConnected();
    
    const subscription = this.connection!.subscribe(this.topics.PLAN);
    
    console.log(`👂 Subscribed to benchmark plans on ${this.topics.PLAN}`);
    
    (async () => {
      for await (const message of subscription) {
        try {
          const plan = this.jsonCodec.decode(message.data) as BenchmarkPlanMessage;
          callback(plan);
        } catch (error) {
          console.error('Failed to decode plan message:', error);
        }
      }
    })();
  }

  /**
   * Subscribe to benchmark run messages for monitoring
   */
  async subscribeRuns(callback: (run: BenchmarkRunMessage) => void): Promise<void> {
    await this.ensureConnected();
    
    const subscription = this.connection!.subscribe(this.topics.RUN);
    
    console.log(`👂 Subscribed to benchmark runs on ${this.topics.RUN}`);
    
    (async () => {
      for await (const message of subscription) {
        try {
          const run = this.jsonCodec.decode(message.data) as BenchmarkRunMessage;
          callback(run);
        } catch (error) {
          console.error('Failed to decode run message:', error);
        }
      }
    })();
  }

  /**
   * Subscribe to benchmark results for monitoring
   */
  async subscribeResults(callback: (result: BenchmarkResultMessage) => void): Promise<void> {
    await this.ensureConnected();
    
    const subscription = this.connection!.subscribe(this.topics.RESULT);
    
    console.log(`👂 Subscribed to benchmark results on ${this.topics.RESULT}`);
    
    (async () => {
      for await (const message of subscription) {
        try {
          const result = this.jsonCodec.decode(message.data) as BenchmarkResultMessage;
          callback(result);
        } catch (error) {
          console.error('Failed to decode result message:', error);
        }
      }
    })();
  }

  /**
   * Publish custom telemetry message to a specific subject
   */
  async publishCustom(subject: string, data: any): Promise<void> {
    await this.ensureConnected();
    
    try {
      const encoded = this.jsonCodec.encode(data);
      await this.connection!.publish(subject, encoded);
      
    } catch (error) {
      console.error(`Failed to publish to ${subject}:`, error);
      throw error;
    }
  }

  /**
   * Publish performance metrics during benchmark execution
   */
  async publishMetrics(traceId: string, metrics: {
    timestamp: string;
    system: string;
    stage: string;
    latency_ms: number;
    candidates_count: number;
    memory_usage_mb?: number;
    cpu_percent?: number;
  }): Promise<void> {
    const subject = `lens.bench.metrics.${traceId}`;
    
    const metricsMessage = {
      trace_id: traceId,
      ...metrics
    };
    
    await this.publishCustom(subject, metricsMessage);
  }

  /**
   * Publish error details during benchmark execution
   */
  async publishError(traceId: string, error: {
    timestamp: string;
    error_type: string;
    message: string;
    stage?: string;
    query_id?: string;
    stack_trace?: string;
  }): Promise<void> {
    const subject = `lens.bench.errors.${traceId}`;
    
    const errorMessage = {
      trace_id: traceId,
      ...error
    };
    
    await this.publishCustom(subject, errorMessage);
    
    console.error(`🚨 Error published - Trace: ${traceId}, Type: ${error.error_type}`);
  }

  /**
   * Publish trace spans for detailed performance analysis
   */
  async publishTrace(traceId: string, span: {
    span_id: string;
    parent_span_id?: string;
    operation_name: string;
    start_time: string;
    end_time: string;
    duration_ms: number;
    tags: Record<string, any>;
    logs?: Array<{
      timestamp: string;
      level: string;
      message: string;
      fields?: Record<string, any>;
    }>;
  }): Promise<void> {
    const subject = `lens.bench.traces.${traceId}`;
    
    const traceMessage = {
      trace_id: traceId,
      ...span
    };
    
    await this.publishCustom(subject, traceMessage);
  }

  /**
   * Health check - verify NATS connection is working
   */
  async healthCheck(): Promise<{ 
    connected: boolean; 
    server?: string; 
    uptime?: number;
  }> {
    if (!this.connection) {
      return { connected: false };
    }

    try {
      // Send a ping to verify connection health
      const rtt = await this.connection.rtt();
      
      return {
        connected: true,
        server: this.connection.getServer(),
        uptime: rtt
      };
      
    } catch (error) {
      console.error('NATS health check failed:', error);
      return { connected: false };
    }
  }

  /**
   * Get connection statistics
   */
  getStats(): {
    connected: boolean;
    server?: string;
    pending: number;
    reconnects: number;
  } {
    if (!this.connection) {
      return {
        connected: false,
        pending: 0,
        reconnects: 0
      };
    }

    const stats = this.connection.stats();
    
    return {
      connected: true,
      server: this.connection.getServer(),
      pending: (stats as any).pending || 0,
      reconnects: (stats as any).reconnects || 0
    };
  }

  /**
   * Gracefully close connection
   */
  async disconnect(): Promise<void> {
    if (this.connection) {
      await this.connection.drain();
      await this.connection.close();
      this.connection = null;
      console.log('📡 NATS connection closed');
    }
  }

  private async ensureConnected(): Promise<void> {
    if (!this.connection) {
      await this.connect();
    }
    
    // Check if connection is still active
    if (this.connection?.isClosed()) {
      console.log('NATS connection was closed, reconnecting...');
      await this.connect();
    }
  }
}

/**
 * Telemetry aggregator for collecting metrics across multiple benchmark runs
 */
export class BenchmarkTelemetryAggregator {
  private telemetry: NATSTelemetry;
  private activeTraces: Map<string, {
    plan?: BenchmarkPlanMessage;
    runs: BenchmarkRunMessage[];
    result?: BenchmarkResultMessage;
    startTime: number;
  }> = new Map();

  constructor(natsUrl: string = 'nats://localhost:4222') {
    this.telemetry = new NATSTelemetry(natsUrl);
  }

  async start(): Promise<void> {
    await this.telemetry.connect();
    
    // Subscribe to all benchmark events
    await this.telemetry.subscribePlans((plan) => {
      this.activeTraces.set(plan.trace_id, {
        plan,
        runs: [],
        startTime: Date.now()
      });
      
      console.log(`🎯 New benchmark planned - ${plan.trace_id}: ${plan.total_queries} queries`);
    });

    await this.telemetry.subscribeRuns((run) => {
      const trace = this.activeTraces.get(run.trace_id);
      if (trace) {
        trace.runs.push(run);
      }
    });

    await this.telemetry.subscribeResults((result) => {
      const trace = this.activeTraces.get(result.trace_id);
      if (trace) {
        trace.result = result;
        
        const duration = Date.now() - trace.startTime;
        const gateStatus = result.promotion_gate_result.passed ? 'PASSED' : 'FAILED';
        
        console.log(`📈 Benchmark completed - ${result.trace_id}: ${gateStatus}, ${duration}ms total`);
        
        // Clean up completed trace after some time
        setTimeout(() => {
          this.activeTraces.delete(result.trace_id);
        }, 60000); // Keep for 1 minute
      }
    });
  }

  /**
   * Get current active benchmark traces
   */
  getActiveTraces(): Array<{
    trace_id: string;
    status: 'planned' | 'running' | 'completed';
    elapsed_ms: number;
    progress?: number;
  }> {
    const now = Date.now();
    
    return Array.from(this.activeTraces.entries()).map(([traceId, trace]) => {
      let status: 'planned' | 'running' | 'completed' = 'planned';
      let progress: number | undefined = undefined;
      
      if (trace.result) {
        status = 'completed';
        progress = 100;
      } else if (trace.runs.length > 0) {
        status = 'running';
        
        // Estimate progress based on run messages
        if (trace.plan) {
          const queryCompletions = trace.runs.filter(r => r.status === 'query_completed').length;
          progress = Math.min(95, (queryCompletions / trace.plan.total_queries) * 100);
        }
      }
      
      return {
        trace_id: traceId,
        status,
        elapsed_ms: now - trace.startTime,
        ...(progress !== undefined && { progress })
      };
    });
  }

  async stop(): Promise<void> {
    await this.telemetry.disconnect();
  }
}