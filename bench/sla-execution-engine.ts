/**
 * SLA-Bounded Execution Engine - Strict 150ms enforcement with fairness
 * 
 * This engine ensures all systems compete under identical SLA constraints
 * with server-side latency counting, client watchdog enforcement, and
 * hardware parity validation.
 */

import { EventEmitter } from 'events';
import * as os from 'os';
import { performance } from 'perf_hooks';
import { CompetitorAdapter, SearchResponse, SystemInfo } from './competitor-adapters';

export interface SLAConfig {
  sla_ms: number;              // Hard SLA limit (default: 150ms)
  watchdog_buffer_ms: number;  // Client-side buffer (default: 20ms)
  hardware_validation: boolean; // Enforce hardware parity
  resource_monitoring: boolean; // Monitor CPU/memory during execution
  timeout_retries: number;     // Retries for timeout handling
}

export interface BenchmarkQuery {
  query_id: string;
  query_text: string;
  suite: string;
  intent: 'exact' | 'identifier' | 'structural' | 'semantic';
  language: string;
  expected_file?: string;  // For validation
}

export interface ExecutionResult {
  query_id: string;
  system_id: string;
  query_text: string;
  suite: string;
  intent: string;
  language: string;
  
  // Timing metrics
  server_latency_ms: number;
  client_latency_ms: number;
  queue_time_ms: number;
  
  // SLA compliance
  within_sla: boolean;
  sla_ms: number;
  timeout_reason?: 'server' | 'client' | 'watchdog';
  
  // Search results
  hits: SearchHit[];
  total_hits: number;
  
  // System state
  cpu_usage_pct: number;
  memory_usage_mb: number;
  concurrent_queries: number;
  
  // Error handling
  error?: string;
  retry_count: number;
  
  // Attestation
  system_info: SystemInfo;
  execution_timestamp: string;
  hardware_fingerprint: string;
}

interface SearchHit {
  file: string;
  line: number;
  column: number;
  snippet: string;
  score: number;
  why_tag: 'exact' | 'struct' | 'semantic' | 'mixed';
  rank: number;
}

interface ResourceSnapshot {
  timestamp: number;
  cpu_usage_pct: number;
  memory_usage_mb: number;
  load_average: number[];
}

/**
 * Hardware attestation and parity enforcement
 */
export class HardwareAttestation {
  private baselineFingerprint?: string;
  private cpuInfo?: any;
  private memInfo?: any;

  /**
   * Collect hardware fingerprint for parity validation
   */
  async collectFingerprint(): Promise<string> {
    const cpuInfo = os.cpus()[0];
    const memInfo = os.totalmem();
    const loadAvg = os.loadavg();
    
    const fingerprint = {
      cpu_model: cpuInfo.model,
      cpu_speed: cpuInfo.speed,
      cpu_cores: os.cpus().length,
      memory_bytes: memInfo,
      architecture: os.arch(),
      platform: os.platform(),
      load_avg: loadAvg[0] // 1-minute load average
    };

    this.cpuInfo = cpuInfo;
    this.memInfo = memInfo;
    
    const fingerprintStr = JSON.stringify(fingerprint);
    return require('crypto').createHash('sha256').update(fingerprintStr).digest('hex').substring(0, 16);
  }

  /**
   * Set baseline fingerprint for comparison
   */
  setBaseline(fingerprint: string): void {
    this.baselineFingerprint = fingerprint;
  }

  /**
   * Validate current hardware matches baseline
   */
  async validateParity(): Promise<{ valid: boolean; reason?: string }> {
    if (!this.baselineFingerprint) {
      return { valid: true }; // No baseline set
    }

    const currentFingerprint = await this.collectFingerprint();
    
    if (currentFingerprint !== this.baselineFingerprint) {
      return {
        valid: false,
        reason: `Hardware fingerprint mismatch. Expected: ${this.baselineFingerprint}, Got: ${currentFingerprint}`
      };
    }

    // Check system load
    const loadAvg = os.loadavg()[0];
    if (loadAvg > os.cpus().length * 0.8) {
      return {
        valid: false,
        reason: `High system load: ${loadAvg.toFixed(2)} (threshold: ${(os.cpus().length * 0.8).toFixed(2)})`
      };
    }

    return { valid: true };
  }

  /**
   * Get detailed hardware information
   */
  getHardwareInfo() {
    return {
      cpu: this.cpuInfo,
      memory_gb: Math.round(this.memInfo / (1024 * 1024 * 1024)),
      load_avg: os.loadavg()
    };
  }
}

/**
 * Resource monitoring during query execution
 */
export class ResourceMonitor {
  private snapshots: ResourceSnapshot[] = [];
  private monitoring = false;
  private intervalId?: NodeJS.Timeout;

  /**
   * Start resource monitoring
   */
  startMonitoring(intervalMs = 50): void {
    if (this.monitoring) return;

    this.monitoring = true;
    this.snapshots = [];

    this.intervalId = setInterval(() => {
      this.snapshots.push({
        timestamp: performance.now(),
        cpu_usage_pct: this.getCpuUsage(),
        memory_usage_mb: process.memoryUsage().rss / (1024 * 1024),
        load_average: os.loadavg()
      });
    }, intervalMs);
  }

  /**
   * Stop monitoring and return statistics
   */
  stopMonitoring(): { avg_cpu: number; max_cpu: number; avg_memory: number; max_memory: number } {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
    
    this.monitoring = false;

    if (this.snapshots.length === 0) {
      return { avg_cpu: 0, max_cpu: 0, avg_memory: 0, max_memory: 0 };
    }

    const cpuValues = this.snapshots.map(s => s.cpu_usage_pct);
    const memoryValues = this.snapshots.map(s => s.memory_usage_mb);

    return {
      avg_cpu: cpuValues.reduce((a, b) => a + b, 0) / cpuValues.length,
      max_cpu: Math.max(...cpuValues),
      avg_memory: memoryValues.reduce((a, b) => a + b, 0) / memoryValues.length,
      max_memory: Math.max(...memoryValues)
    };
  }

  private getCpuUsage(): number {
    // Simple CPU usage approximation based on load average
    const loadAvg = os.loadavg()[0];
    const cores = os.cpus().length;
    return Math.min((loadAvg / cores) * 100, 100);
  }
}

/**
 * SLA-bounded query execution with strict enforcement
 */
export class SLAExecutionEngine extends EventEmitter {
  private config: SLAConfig;
  private hardwareAttestation: HardwareAttestation;
  private activeQueries = new Map<string, { startTime: number; timeout: NodeJS.Timeout }>();
  private executionStats = {
    total_queries: 0,
    timeouts: 0,
    errors: 0,
    sla_violations: 0
  };

  constructor(config: Partial<SLAConfig> = {}) {
    super();
    
    this.config = {
      sla_ms: 150,
      watchdog_buffer_ms: 20,
      hardware_validation: true,
      resource_monitoring: true,
      timeout_retries: 1,
      ...config
    };

    this.hardwareAttestation = new HardwareAttestation();
  }

  /**
   * Initialize engine with hardware attestation
   */
  async initialize(): Promise<void> {
    if (this.config.hardware_validation) {
      const fingerprint = await this.hardwareAttestation.collectFingerprint();
      this.hardwareAttestation.setBaseline(fingerprint);
      
      console.log(`🔒 Hardware baseline established: ${fingerprint}`);
      console.log(`💻 Hardware info:`, this.hardwareAttestation.getHardwareInfo());
    }
  }

  /**
   * Execute query with strict SLA enforcement
   */
  async executeQuery(
    adapter: CompetitorAdapter,
    query: BenchmarkQuery
  ): Promise<ExecutionResult> {
    // Pre-execution validation
    if (this.config.hardware_validation) {
      const parityCheck = await this.hardwareAttestation.validateParity();
      if (!parityCheck.valid) {
        throw new Error(`Hardware parity violation: ${parityCheck.reason}`);
      }
    }

    const queryId = `${query.query_id}_${Date.now()}`;
    const queueStartTime = performance.now();
    
    // Resource monitoring
    const resourceMonitor = new ResourceMonitor();
    if (this.config.resource_monitoring) {
      resourceMonitor.startMonitoring();
    }

    let result: ExecutionResult;
    let retryCount = 0;

    while (retryCount <= this.config.timeout_retries) {
      try {
        result = await this.executeSingleQuery(adapter, query, queryId, queueStartTime, resourceMonitor);
        break;
      } catch (error) {
        retryCount++;
        if (retryCount > this.config.timeout_retries) {
          result = this.createErrorResult(adapter, query, queryId, queueStartTime, error, retryCount - 1);
          break;
        }
        
        console.warn(`⚠️  Query ${query.query_id} attempt ${retryCount} failed: ${error.message}`);
        
        // Brief backoff before retry
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }

    // Stop resource monitoring
    const resourceStats = resourceMonitor.stopMonitoring();
    result.cpu_usage_pct = resourceStats.avg_cpu;
    result.memory_usage_mb = resourceStats.avg_memory;

    // Update execution statistics
    this.updateExecutionStats(result);

    // Emit events for monitoring
    this.emit('queryComplete', result);
    
    if (!result.within_sla) {
      this.emit('slaViolation', result);
    }

    return result;
  }

  /**
   * Execute single query attempt with watchdog enforcement
   */
  private async executeSingleQuery(
    adapter: CompetitorAdapter,
    query: BenchmarkQuery,
    queryId: string,
    queueStartTime: number,
    resourceMonitor: ResourceMonitor
  ): Promise<ExecutionResult> {
    const serverStartTime = performance.now();
    const queueTime = serverStartTime - queueStartTime;
    
    // Set up client-side watchdog
    const watchdogTimeout = this.config.sla_ms + this.config.watchdog_buffer_ms;
    let timedOut = false;
    let timeoutReason: 'server' | 'client' | 'watchdog' = 'server';

    const watchdogPromise = new Promise<never>((_, reject) => {
      const timeout = setTimeout(() => {
        timedOut = true;
        timeoutReason = 'watchdog';
        reject(new Error(`Watchdog timeout after ${watchdogTimeout}ms`));
      }, watchdogTimeout);

      this.activeQueries.set(queryId, { startTime: serverStartTime, timeout });
    });

    try {
      // Execute search with race between actual search and watchdog
      const searchPromise = adapter.search(query.query_text, this.config.sla_ms);
      const response = await Promise.race([searchPromise, watchdogPromise]) as SearchResponse;

      const serverLatency = performance.now() - serverStartTime;
      const clientLatency = performance.now() - queueStartTime;

      // Clean up watchdog
      this.cleanupQuery(queryId);

      // Determine SLA compliance
      const withinSla = serverLatency <= this.config.sla_ms && response.within_sla;
      
      if (response.error && !withinSla) {
        timeoutReason = response.latency_ms > this.config.sla_ms ? 'server' : 'client';
      }

      // Build execution result
      const result: ExecutionResult = {
        query_id: query.query_id,
        system_id: adapter.getSystemInfo ? (await adapter.getSystemInfo()).system_id : 'unknown',
        query_text: query.query_text,
        suite: query.suite,
        intent: query.intent,
        language: query.language,

        server_latency_ms: Math.round(serverLatency * 100) / 100,
        client_latency_ms: Math.round(clientLatency * 100) / 100,
        queue_time_ms: Math.round(queueTime * 100) / 100,

        within_sla: withinSla,
        sla_ms: this.config.sla_ms,
        timeout_reason: withinSla ? undefined : timeoutReason,

        hits: this.normalizeHits(response.hits || []),
        total_hits: response.hits?.length || 0,

        cpu_usage_pct: 0, // Will be set by caller
        memory_usage_mb: 0, // Will be set by caller  
        concurrent_queries: this.activeQueries.size,

        error: response.error,
        retry_count: 0, // Will be set by caller

        system_info: response.system_info || await adapter.getSystemInfo(),
        execution_timestamp: new Date().toISOString(),
        hardware_fingerprint: await this.hardwareAttestation.collectFingerprint()
      };

      return result;

    } catch (error) {
      this.cleanupQuery(queryId);
      
      if (timedOut) {
        const latency = performance.now() - serverStartTime;
        throw new Error(`Query timeout: ${latency.toFixed(2)}ms (SLA: ${this.config.sla_ms}ms, reason: ${timeoutReason})`);
      }
      
      throw error;
    }
  }

  /**
   * Create error result for failed queries
   */
  private createErrorResult(
    adapter: CompetitorAdapter,
    query: BenchmarkQuery,
    queryId: string,
    queueStartTime: number,
    error: Error,
    retryCount: number
  ): ExecutionResult {
    const now = performance.now();
    
    return {
      query_id: query.query_id,
      system_id: adapter.constructor.name,
      query_text: query.query_text,
      suite: query.suite,
      intent: query.intent,
      language: query.language,

      server_latency_ms: now - queueStartTime,
      client_latency_ms: now - queueStartTime,
      queue_time_ms: 0,

      within_sla: false,
      sla_ms: this.config.sla_ms,
      timeout_reason: 'server',

      hits: [],
      total_hits: 0,

      cpu_usage_pct: 0,
      memory_usage_mb: 0,
      concurrent_queries: this.activeQueries.size,

      error: error.message,
      retry_count: retryCount,

      system_info: { system_id: 'unknown', version: 'unknown', config_fingerprint: 'error', hardware_info: {} as any },
      execution_timestamp: new Date().toISOString(),
      hardware_fingerprint: 'error'
    };
  }

  /**
   * Normalize search hits with ranking
   */
  private normalizeHits(hits: any[]): SearchHit[] {
    return hits.map((hit, index) => ({
      file: hit.file || '',
      line: hit.line || 1,
      column: hit.column || 1,
      snippet: hit.snippet || '',
      score: hit.score || (1.0 - index * 0.01),
      why_tag: hit.why_tag || 'mixed',
      rank: index + 1
    }));
  }

  /**
   * Clean up query tracking
   */
  private cleanupQuery(queryId: string): void {
    const query = this.activeQueries.get(queryId);
    if (query) {
      clearTimeout(query.timeout);
      this.activeQueries.delete(queryId);
    }
  }

  /**
   * Update execution statistics
   */
  private updateExecutionStats(result: ExecutionResult): void {
    this.executionStats.total_queries++;
    
    if (result.error) {
      this.executionStats.errors++;
    }
    
    if (!result.within_sla) {
      this.executionStats.sla_violations++;
      
      if (result.timeout_reason) {
        this.executionStats.timeouts++;
      }
    }
  }

  /**
   * Get execution statistics
   */
  getExecutionStats() {
    return {
      ...this.executionStats,
      sla_compliance_rate: this.executionStats.total_queries > 0 
        ? (this.executionStats.total_queries - this.executionStats.sla_violations) / this.executionStats.total_queries
        : 0,
      error_rate: this.executionStats.total_queries > 0
        ? this.executionStats.errors / this.executionStats.total_queries  
        : 0,
      timeout_rate: this.executionStats.total_queries > 0
        ? this.executionStats.timeouts / this.executionStats.total_queries
        : 0
    };
  }

  /**
   * Emergency shutdown - cancel all active queries
   */
  async emergencyShutdown(): Promise<void> {
    console.log(`🚨 Emergency shutdown: cancelling ${this.activeQueries.size} active queries`);
    
    for (const [queryId, query] of this.activeQueries) {
      clearTimeout(query.timeout);
    }
    
    this.activeQueries.clear();
    this.emit('emergencyShutdown', this.executionStats);
  }

  /**
   * Validate system meets minimum requirements for benchmarking
   */
  async validateSystemRequirements(): Promise<{ valid: boolean; warnings: string[] }> {
    const warnings: string[] = [];
    let valid = true;

    // Check CPU
    const cpus = os.cpus();
    if (cpus.length < 4) {
      warnings.push(`Low CPU core count: ${cpus.length} (recommended: 4+)`);
    }

    // Check memory
    const totalMemGB = os.totalmem() / (1024 * 1024 * 1024);
    if (totalMemGB < 8) {
      warnings.push(`Low memory: ${totalMemGB.toFixed(1)}GB (recommended: 8GB+)`);
      valid = false;
    }

    // Check load average
    const loadAvg = os.loadavg()[0];
    const loadThreshold = cpus.length * 0.8;
    if (loadAvg > loadThreshold) {
      warnings.push(`High system load: ${loadAvg.toFixed(2)} (threshold: ${loadThreshold.toFixed(2)})`);
    }

    // Check available memory
    const freeMemGB = os.freemem() / (1024 * 1024 * 1024);
    if (freeMemGB < 2) {
      warnings.push(`Low free memory: ${freeMemGB.toFixed(1)}GB (recommended: 2GB+)`);
    }

    return { valid, warnings };
  }
}

/**
 * Batch execution coordinator for running multiple queries across systems
 */
export class BatchExecutor {
  private engine: SLAExecutionEngine;
  private adapters: Map<string, CompetitorAdapter>;
  private results: ExecutionResult[] = [];

  constructor(
    engine: SLAExecutionEngine,
    adapters: Map<string, CompetitorAdapter>
  ) {
    this.engine = engine;
    this.adapters = adapters;
  }

  /**
   * Execute batch of queries across all systems
   */
  async executeBatch(queries: BenchmarkQuery[]): Promise<ExecutionResult[]> {
    console.log(`🚀 Starting batch execution: ${queries.length} queries × ${this.adapters.size} systems`);
    
    const results: ExecutionResult[] = [];
    let completedQueries = 0;
    
    // Execute queries in parallel across systems
    const promises: Promise<ExecutionResult>[] = [];

    for (const query of queries) {
      for (const [systemId, adapter] of this.adapters) {
        const promise = this.engine.executeQuery(adapter, query)
          .then(result => {
            completedQueries++;
            const progress = (completedQueries / (queries.length * this.adapters.size)) * 100;
            console.log(`📊 Progress: ${completedQueries}/${queries.length * this.adapters.size} (${progress.toFixed(1)}%)`);
            return result;
          })
          .catch(error => {
            console.error(`❌ Query ${query.query_id} failed on ${systemId}: ${error.message}`);
            return this.createFailureResult(query, systemId, error);
          });

        promises.push(promise);
      }
    }

    // Wait for all executions to complete
    const batchResults = await Promise.all(promises);
    results.push(...batchResults);

    // Log final statistics
    const stats = this.engine.getExecutionStats();
    console.log(`✅ Batch execution complete:`);
    console.log(`   Total queries: ${stats.total_queries}`);
    console.log(`   SLA compliance: ${(stats.sla_compliance_rate * 100).toFixed(1)}%`);
    console.log(`   Error rate: ${(stats.error_rate * 100).toFixed(1)}%`);
    console.log(`   Timeout rate: ${(stats.timeout_rate * 100).toFixed(1)}%`);

    this.results = results;
    return results;
  }

  private createFailureResult(query: BenchmarkQuery, systemId: string, error: Error): ExecutionResult {
    return {
      query_id: query.query_id,
      system_id: systemId,
      query_text: query.query_text,
      suite: query.suite,
      intent: query.intent,
      language: query.language,
      server_latency_ms: 0,
      client_latency_ms: 0,
      queue_time_ms: 0,
      within_sla: false,
      sla_ms: 150,
      hits: [],
      total_hits: 0,
      cpu_usage_pct: 0,
      memory_usage_mb: 0,
      concurrent_queries: 0,
      error: error.message,
      retry_count: 0,
      system_info: { system_id: systemId, version: 'unknown', config_fingerprint: 'error', hardware_info: {} as any },
      execution_timestamp: new Date().toISOString(),
      hardware_fingerprint: 'error'
    };
  }

  /**
   * Get execution results
   */
  getResults(): ExecutionResult[] {
    return this.results;
  }
}