/**
 * Load Test Orchestrator for Gemma Variants
 * 
 * Manages complex load testing scenarios including:
 * - QPS vs Latency curve generation
 * - Concurrent user simulation
 * - Resource utilization monitoring
 * - Breaking point analysis
 * - Real-time metrics collection
 */

import { z } from 'zod';
import { performance } from 'perf_hooks';
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { EventEmitter } from 'events';
import * as os from 'os';
import * as fs from 'fs';
import * as path from 'path';

// Schema definitions for load testing

const LoadTestScenarioSchema = z.object({
  name: z.string(),
  description: z.string(),
  targetQPS: z.array(z.number()),
  concurrentUsers: z.array(z.number()),
  durationSeconds: z.number().default(300),
  warmupSeconds: z.number().default(60),
  cooldownSeconds: z.number().default(30),
  requestTimeoutMs: z.number().default(30000),
  maxErrors: z.number().default(100),
  queries: z.array(z.string()).min(1)
});

const ResourceMonitoringConfigSchema = z.object({
  enabled: z.boolean().default(true),
  intervalMs: z.number().default(1000),
  metrics: z.array(z.enum(['cpu', 'memory', 'disk', 'network'])).default(['cpu', 'memory']),
  thresholds: z.object({
    cpuPercent: z.number().default(95),
    memoryPercent: z.number().default(90),
    diskIOPS: z.number().default(1000)
  }).default({})
});

const LoadTestConfigSchema = z.object({
  variant: z.string(),
  scenarios: z.array(LoadTestScenarioSchema),
  resourceMonitoring: ResourceMonitoringConfigSchema.default({}),
  outputDir: z.string(),
  realTimeMetrics: z.boolean().default(true),
  saveRawData: z.boolean().default(true)
});

const RequestMetricsSchema = z.object({
  timestamp: z.number(),
  latency: z.number(),
  success: z.boolean(),
  errorCode: z.string().optional(),
  responseSize: z.number().optional(),
  workerId: z.string()
});

const ResourceMetricsSchema = z.object({
  timestamp: z.number(),
  cpu: z.object({
    usage: z.number(),
    load1m: z.number(),
    loadavg: z.array(z.number())
  }),
  memory: z.object({
    used: z.number(),
    free: z.number(),
    total: z.number(),
    heapUsed: z.number()
  }),
  disk: z.object({
    readIOPS: z.number(),
    writeIOPS: z.number()
  }).optional(),
  network: z.object({
    bytesIn: z.number(),
    bytesOut: z.number(),
    packetsIn: z.number(),
    packetsOut: z.number()
  }).optional()
});

const LoadTestResultSchema = z.object({
  scenario: z.string(),
  variant: z.string(),
  startTime: z.string(),
  endTime: z.string(),
  durationMs: z.number(),
  totalRequests: z.number(),
  successfulRequests: z.number(),
  failedRequests: z.number(),
  actualQPS: z.number(),
  latencyStats: z.object({
    min: z.number(),
    max: z.number(),
    mean: z.number(),
    median: z.number(),
    p95: z.number(),
    p99: z.number(),
    p999: z.number(),
    stdDev: z.number()
  }),
  errorDistribution: z.record(z.string(), z.number()),
  resourceUtilization: z.object({
    peakCPU: z.number(),
    avgCPU: z.number(),
    peakMemory: z.number(),
    avgMemory: z.number()
  }),
  qpsLatencyCurve: z.array(z.object({
    qps: z.number(),
    p50Latency: z.number(),
    p95Latency: z.number(),
    errorRate: z.number()
  })),
  breakingPoint: z.object({
    maxQPS: z.number(),
    maxConcurrentUsers: z.number(),
    firstErrorQPS: z.number(),
    degradationQPS: z.number()
  })
});

export type LoadTestScenario = z.infer<typeof LoadTestScenarioSchema>;
export type ResourceMonitoringConfig = z.infer<typeof ResourceMonitoringConfigSchema>;
export type LoadTestConfig = z.infer<typeof LoadTestConfigSchema>;
export type RequestMetrics = z.infer<typeof RequestMetricsSchema>;
export type ResourceMetrics = z.infer<typeof ResourceMetricsSchema>;
export type LoadTestResult = z.infer<typeof LoadTestResultSchema>;

/**
 * Main load test orchestrator class
 */
export class LoadTestOrchestrator extends EventEmitter {
  private workers: Worker[] = [];
  private resourceMonitor?: NodeJS.Timeout;
  private metricsCollector: MetricsCollector;
  private isRunning = false;
  
  constructor() {
    super();
    this.metricsCollector = new MetricsCollector();
  }

  /**
   * Execute comprehensive load testing for a variant
   */
  async executeLoadTest(config: LoadTestConfig): Promise<LoadTestResult[]> {
    console.log(`🚀 Starting load test for ${config.variant}`);
    
    this.validateConfig(config);
    await this.setupOutputDirectory(config.outputDir);
    
    const results: LoadTestResult[] = [];
    this.isRunning = true;
    
    try {
      // Start resource monitoring
      if (config.resourceMonitoring.enabled) {
        await this.startResourceMonitoring(config.resourceMonitoring);
      }
      
      // Execute each scenario
      for (const scenario of config.scenarios) {
        console.log(`📊 Executing scenario: ${scenario.name}`);
        
        const result = await this.executeScenario(config.variant, scenario, config);
        results.push(result);
        
        // Save intermediate results
        if (config.saveRawData) {
          await this.saveScenarioResult(result, config.outputDir);
        }
        
        this.emit('scenarioComplete', result);
        
        // Cooldown between scenarios
        if (scenario.cooldownSeconds > 0) {
          console.log(`⏸️ Cooldown period: ${scenario.cooldownSeconds}s`);
          await this.sleep(scenario.cooldownSeconds * 1000);
        }
      }
      
    } finally {
      this.isRunning = false;
      await this.cleanup();
    }
    
    console.log(`✅ Load test completed for ${config.variant}`);
    return results;
  }

  /**
   * Execute a single load test scenario
   */
  private async executeScenario(
    variant: string,
    scenario: LoadTestScenario,
    config: LoadTestConfig
  ): Promise<LoadTestResult> {
    
    const startTime = new Date();
    const requestMetrics: RequestMetrics[] = [];
    const resourceMetrics: ResourceMetrics[] = [];
    
    // Warmup phase
    if (scenario.warmupSeconds > 0) {
      console.log(`  🔥 Warmup phase: ${scenario.warmupSeconds}s`);
      await this.runWarmupPhase(variant, scenario);
    }
    
    // Main load test execution
    console.log(`  🎯 Main load test: ${scenario.durationSeconds}s`);
    
    const qpsLatencyCurve: Array<{
      qps: number;
      p50Latency: number;
      p95Latency: number;
      errorRate: number;
    }> = [];
    
    let breakingPoint = {
      maxQPS: 0,
      maxConcurrentUsers: 0,
      firstErrorQPS: 0,
      degradationQPS: 0
    };
    
    // Test different QPS levels to build curve
    for (const targetQPS of scenario.targetQPS) {
      console.log(`    📈 Testing QPS: ${targetQPS}`);
      
      const qpsResult = await this.executeQPSTest(
        variant,
        scenario,
        targetQPS,
        config
      );
      
      requestMetrics.push(...qpsResult.requests);
      resourceMetrics.push(...qpsResult.resources);
      
      qpsLatencyCurve.push({
        qps: qpsResult.actualQPS,
        p50Latency: qpsResult.p50Latency,
        p95Latency: qpsResult.p95Latency,
        errorRate: qpsResult.errorRate
      });
      
      // Update breaking point analysis
      breakingPoint.maxQPS = Math.max(breakingPoint.maxQPS, qpsResult.actualQPS);
      
      if (qpsResult.errorRate > 0 && breakingPoint.firstErrorQPS === 0) {
        breakingPoint.firstErrorQPS = qpsResult.actualQPS;
      }
      
      if (qpsResult.errorRate > 5 && breakingPoint.degradationQPS === 0) {
        breakingPoint.degradationQPS = qpsResult.actualQPS;
      }
      
      // Stop if error rate becomes unacceptable
      if (qpsResult.errorRate > 50) {
        console.log(`    🛑 Stopping due to high error rate: ${qpsResult.errorRate}%`);
        break;
      }
    }
    
    // Test different concurrency levels
    for (const concurrentUsers of scenario.concurrentUsers) {
      console.log(`    👥 Testing concurrent users: ${concurrentUsers}`);
      
      const concurrencyResult = await this.executeConcurrencyTest(
        variant,
        scenario,
        concurrentUsers,
        config
      );
      
      requestMetrics.push(...concurrencyResult.requests);
      resourceMetrics.push(...concurrencyResult.resources);
      
      breakingPoint.maxConcurrentUsers = Math.max(
        breakingPoint.maxConcurrentUsers,
        concurrencyResult.maxHandledUsers
      );
    }
    
    const endTime = new Date();
    const durationMs = endTime.getTime() - startTime.getTime();
    
    // Calculate aggregate statistics
    const successfulRequests = requestMetrics.filter(r => r.success).length;
    const failedRequests = requestMetrics.length - successfulRequests;
    const actualQPS = requestMetrics.length / (durationMs / 1000);
    
    const latencies = requestMetrics.filter(r => r.success).map(r => r.latency);
    const latencyStats = this.calculateLatencyStats(latencies);
    
    const errorDistribution = this.calculateErrorDistribution(
      requestMetrics.filter(r => !r.success)
    );
    
    const resourceUtilization = this.calculateResourceUtilization(resourceMetrics);
    
    return LoadTestResultSchema.parse({
      scenario: scenario.name,
      variant,
      startTime: startTime.toISOString(),
      endTime: endTime.toISOString(),
      durationMs,
      totalRequests: requestMetrics.length,
      successfulRequests,
      failedRequests,
      actualQPS,
      latencyStats,
      errorDistribution,
      resourceUtilization,
      qpsLatencyCurve,
      breakingPoint
    });
  }

  /**
   * Execute QPS-focused load test
   */
  private async executeQPSTest(
    variant: string,
    scenario: LoadTestScenario,
    targetQPS: number,
    config: LoadTestConfig
  ): Promise<{
    actualQPS: number;
    p50Latency: number;
    p95Latency: number;
    errorRate: number;
    requests: RequestMetrics[];
    resources: ResourceMetrics[];
  }> {
    
    const testDurationMs = 30000; // 30 seconds for QPS test
    const requestInterval = 1000 / targetQPS; // ms between requests
    
    const requests: RequestMetrics[] = [];
    const resources: ResourceMetrics[] = [];
    
    const resourceCollector = setInterval(() => {
      resources.push(this.collectResourceMetrics());
    }, 1000);
    
    const startTime = performance.now();
    const workers = this.createWorkers(Math.min(targetQPS / 10, 50)); // Dynamic worker count
    
    let requestCount = 0;
    const requestTimer = setInterval(async () => {
      if (performance.now() - startTime > testDurationMs) {
        clearInterval(requestTimer);
        return;
      }
      
      const workerId = `worker-${requestCount % workers.length}`;
      const request = this.sendWorkerRequest(workerId, variant, scenario.queries);
      
      request.then(result => {
        requests.push({
          timestamp: performance.now(),
          latency: result.latency,
          success: result.success,
          errorCode: result.errorCode,
          responseSize: result.responseSize,
          workerId
        });
      });
      
      requestCount++;
    }, requestInterval);
    
    // Wait for test completion
    await new Promise(resolve => setTimeout(resolve, testDurationMs + 5000));
    
    clearInterval(resourceCollector);
    await this.terminateWorkers(workers);
    
    const actualQPS = requests.length / (testDurationMs / 1000);
    const successfulRequests = requests.filter(r => r.success);
    const latencies = successfulRequests.map(r => r.latency);
    
    return {
      actualQPS,
      p50Latency: this.percentile(latencies, 0.5),
      p95Latency: this.percentile(latencies, 0.95),
      errorRate: ((requests.length - successfulRequests.length) / requests.length) * 100,
      requests,
      resources
    };
  }

  /**
   * Execute concurrency-focused load test
   */
  private async executeConcurrencyTest(
    variant: string,
    scenario: LoadTestScenario,
    concurrentUsers: number,
    config: LoadTestConfig
  ): Promise<{
    maxHandledUsers: number;
    requests: RequestMetrics[];
    resources: ResourceMetrics[];
  }> {
    
    const testDurationMs = 60000; // 1 minute for concurrency test
    const requests: RequestMetrics[] = [];
    const resources: ResourceMetrics[] = [];
    
    const resourceCollector = setInterval(() => {
      resources.push(this.collectResourceMetrics());
    }, 1000);
    
    const workers = this.createWorkers(concurrentUsers);
    const workerPromises: Promise<void>[] = [];
    
    // Start all workers
    for (let i = 0; i < concurrentUsers; i++) {
      const workerId = `concurrent-worker-${i}`;
      
      const workerTask = this.runContinuousWorker(
        workerId,
        variant,
        scenario.queries,
        testDurationMs,
        (result) => {
          requests.push({
            timestamp: performance.now(),
            latency: result.latency,
            success: result.success,
            errorCode: result.errorCode,
            responseSize: result.responseSize,
            workerId
          });
        }
      );
      
      workerPromises.push(workerTask);
    }
    
    // Wait for all workers to complete
    await Promise.all(workerPromises);
    
    clearInterval(resourceCollector);
    await this.terminateWorkers(workers);
    
    // Determine max handled users (based on error rate)
    const errorRate = ((requests.length - requests.filter(r => r.success).length) / requests.length) * 100;
    const maxHandledUsers = errorRate < 5 ? concurrentUsers : Math.floor(concurrentUsers * 0.8);
    
    return {
      maxHandledUsers,
      requests,
      resources
    };
  }

  /**
   * Create worker threads for load testing
   */
  private createWorkers(count: number): Worker[] {
    const workers: Worker[] = [];
    
    for (let i = 0; i < count; i++) {
      const worker = new Worker(__filename, {
        workerData: { isWorker: true, workerId: `worker-${i}` }
      });
      
      workers.push(worker);
    }
    
    return workers;
  }

  /**
   * Send a request to a worker thread
   */
  private async sendWorkerRequest(
    workerId: string,
    variant: string,
    queries: string[]
  ): Promise<{
    latency: number;
    success: boolean;
    errorCode?: string;
    responseSize?: number;
  }> {
    // Simulate request processing
    const start = performance.now();
    
    try {
      // Simulate variable latency based on variant
      const baseLatency = variant === 'gemma-768' ? 50 : 25;
      const jitter = Math.random() * 20;
      const simulatedLatency = baseLatency + jitter;
      
      await new Promise(resolve => setTimeout(resolve, simulatedLatency));
      
      const latency = performance.now() - start;
      const success = Math.random() > 0.02; // 2% error rate simulation
      
      return {
        latency,
        success,
        errorCode: success ? undefined : 'TIMEOUT',
        responseSize: 1024 + Math.floor(Math.random() * 512)
      };
      
    } catch (error) {
      return {
        latency: performance.now() - start,
        success: false,
        errorCode: 'ERROR',
        responseSize: 0
      };
    }
  }

  /**
   * Run a continuous worker for concurrency testing
   */
  private async runContinuousWorker(
    workerId: string,
    variant: string,
    queries: string[],
    durationMs: number,
    callback: (result: any) => void
  ): Promise<void> {
    const endTime = performance.now() + durationMs;
    
    while (performance.now() < endTime && this.isRunning) {
      const result = await this.sendWorkerRequest(workerId, variant, queries);
      callback(result);
      
      // Small delay between requests
      await new Promise(resolve => setTimeout(resolve, 10 + Math.random() * 40));
    }
  }

  /**
   * Terminate worker threads
   */
  private async terminateWorkers(workers: Worker[]): Promise<void> {
    await Promise.all(workers.map(worker => worker.terminate()));
  }

  /**
   * Collect current resource metrics
   */
  private collectResourceMetrics(): ResourceMetrics {
    const memUsage = process.memoryUsage();
    const loadAvg = os.loadavg();
    
    return ResourceMetricsSchema.parse({
      timestamp: performance.now(),
      cpu: {
        usage: 0, // Would use actual CPU monitoring library
        load1m: loadAvg[0],
        loadavg: loadAvg
      },
      memory: {
        used: memUsage.heapUsed,
        free: os.freemem(),
        total: os.totalmem(),
        heapUsed: memUsage.heapUsed
      }
    });
  }

  /**
   * Start resource monitoring
   */
  private async startResourceMonitoring(config: ResourceMonitoringConfig): Promise<void> {
    this.resourceMonitor = setInterval(() => {
      const metrics = this.collectResourceMetrics();
      this.emit('resourceMetrics', metrics);
      
      // Check thresholds
      if (metrics.cpu.usage > config.thresholds.cpuPercent) {
        this.emit('resourceThresholdExceeded', 'cpu', metrics.cpu.usage);
      }
      
      const memoryPercent = (metrics.memory.used / metrics.memory.total) * 100;
      if (memoryPercent > config.thresholds.memoryPercent) {
        this.emit('resourceThresholdExceeded', 'memory', memoryPercent);
      }
      
    }, config.intervalMs);
  }

  /**
   * Run warmup phase
   */
  private async runWarmupPhase(variant: string, scenario: LoadTestScenario): Promise<void> {
    const warmupRequests = Math.min(100, scenario.queries.length * 5);
    
    for (let i = 0; i < warmupRequests; i++) {
      await this.sendWorkerRequest(
        'warmup-worker',
        variant,
        scenario.queries
      );
      
      if (i % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
  }

  // Statistical calculation methods

  private calculateLatencyStats(latencies: number[]): any {
    if (latencies.length === 0) {
      return {
        min: 0, max: 0, mean: 0, median: 0,
        p95: 0, p99: 0, p999: 0, stdDev: 0
      };
    }
    
    const sorted = [...latencies].sort((a, b) => a - b);
    const mean = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    const variance = latencies.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / latencies.length;
    
    return {
      min: sorted[0],
      max: sorted[sorted.length - 1],
      mean,
      median: this.percentile(sorted, 0.5),
      p95: this.percentile(sorted, 0.95),
      p99: this.percentile(sorted, 0.99),
      p999: this.percentile(sorted, 0.999),
      stdDev: Math.sqrt(variance)
    };
  }

  private calculateErrorDistribution(failedRequests: RequestMetrics[]): Record<string, number> {
    const distribution: Record<string, number> = {};
    
    for (const request of failedRequests) {
      const errorCode = request.errorCode || 'UNKNOWN';
      distribution[errorCode] = (distribution[errorCode] || 0) + 1;
    }
    
    return distribution;
  }

  private calculateResourceUtilization(metrics: ResourceMetrics[]): any {
    if (metrics.length === 0) {
      return { peakCPU: 0, avgCPU: 0, peakMemory: 0, avgMemory: 0 };
    }
    
    const cpuValues = metrics.map(m => m.cpu.usage);
    const memoryValues = metrics.map(m => (m.memory.used / m.memory.total) * 100);
    
    return {
      peakCPU: Math.max(...cpuValues),
      avgCPU: cpuValues.reduce((a, b) => a + b, 0) / cpuValues.length,
      peakMemory: Math.max(...memoryValues),
      avgMemory: memoryValues.reduce((a, b) => a + b, 0) / memoryValues.length
    };
  }

  private percentile(values: number[], p: number): number {
    if (values.length === 0) return 0;
    const index = p * (values.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index - lower;
    
    if (lower === upper) {
      return values[lower] || 0;
    }
    
    return ((values[lower] || 0) * (1 - weight)) + ((values[upper] || 0) * weight);
  }

  // Utility methods

  private validateConfig(config: LoadTestConfig): void {
    const result = LoadTestConfigSchema.safeParse(config);
    if (!result.success) {
      throw new Error(`Invalid load test configuration: ${result.error.message}`);
    }
  }

  private async setupOutputDirectory(outputDir: string): Promise<void> {
    await fs.promises.mkdir(outputDir, { recursive: true });
  }

  private async saveScenarioResult(result: LoadTestResult, outputDir: string): Promise<void> {
    const filename = `${result.scenario}_${result.variant}_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    const filepath = path.join(outputDir, filename);
    
    await fs.promises.writeFile(
      filepath,
      JSON.stringify(result, null, 2),
      'utf8'
    );
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private async cleanup(): Promise<void> {
    if (this.resourceMonitor) {
      clearInterval(this.resourceMonitor);
    }
    
    await this.terminateWorkers(this.workers);
    this.workers = [];
  }
}

/**
 * Metrics collector for real-time monitoring
 */
class MetricsCollector {
  private metrics: Map<string, any[]> = new Map();
  
  collect(type: string, data: any): void {
    if (!this.metrics.has(type)) {
      this.metrics.set(type, []);
    }
    
    this.metrics.get(type)!.push({
      timestamp: performance.now(),
      data
    });
  }
  
  getMetrics(type: string): any[] {
    return this.metrics.get(type) || [];
  }
  
  clear(): void {
    this.metrics.clear();
  }
  
  export(): Record<string, any[]> {
    return Object.fromEntries(this.metrics);
  }
}

/**
 * Worker thread implementation for load testing
 */
if (!isMainThread && workerData?.isWorker) {
  // Worker thread code would be implemented here
  // This handles the actual request execution in isolated threads
  
  parentPort?.on('message', async (message) => {
    const { type, payload } = message;
    
    switch (type) {
      case 'executeRequest':
        try {
          const result = await executeLoadTestRequest(payload);
          parentPort?.postMessage({ type: 'requestComplete', result });
        } catch (error) {
          parentPort?.postMessage({ 
            type: 'requestError', 
            error: error instanceof Error ? error.message : 'Unknown error' 
          });
        }
        break;
        
      case 'shutdown':
        process.exit(0);
        break;
    }
  });
  
  async function executeLoadTestRequest(payload: any): Promise<any> {
    // Implement actual request execution logic here
    // This would call the actual Gemma variant endpoint
    
    const start = performance.now();
    
    // Simulate request processing
    await new Promise(resolve => setTimeout(resolve, 10 + Math.random() * 40));
    
    return {
      latency: performance.now() - start,
      success: true,
      responseSize: 1024
    };
  }
}