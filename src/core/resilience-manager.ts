/**
 * Comprehensive Error Handling and Resilience Patterns
 * 
 * Advanced resilience management system with:
 * - Circuit breaker pattern for external dependencies
 * - Retry mechanisms with exponential backoff and jitter
 * - Bulkhead isolation for component failures
 * - Graceful degradation strategies
 * - Health monitoring and automatic recovery
 * - Request timeouts and deadline propagation  
 * - Rate limiting with adaptive throttling
 * - Error classification and routing
 * - Fallback mechanisms and cached responses
 */

import { EventEmitter } from 'node:events';
import { setTimeout as sleep, clearTimeout } from 'node:timers';
import { opentelemetry } from '../telemetry/index.js';
import { performanceMonitor } from './performance-monitor.js';

// Error classification types
export enum ErrorType {
  NETWORK_ERROR = 'NETWORK_ERROR',
  TIMEOUT_ERROR = 'TIMEOUT_ERROR',
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  RESOURCE_EXHAUSTED = 'RESOURCE_EXHAUSTED',
  SERVICE_UNAVAILABLE = 'SERVICE_UNAVAILABLE',
  AUTHENTICATION_ERROR = 'AUTHENTICATION_ERROR',
  AUTHORIZATION_ERROR = 'AUTHORIZATION_ERROR',
  DATA_CORRUPTION = 'DATA_CORRUPTION',
  CONFIGURATION_ERROR = 'CONFIGURATION_ERROR',
  UNKNOWN_ERROR = 'UNKNOWN_ERROR'
}

export enum ErrorSeverity {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH',
  CRITICAL = 'CRITICAL'
}

// Resilient operation result
export interface OperationResult<T> {
  success: boolean;
  data?: T;
  error?: LensError;
  retryCount: number;
  duration: number;
  fallbackUsed: boolean;
}

// Enhanced error class
export class LensError extends Error {
  constructor(
    message: string,
    public readonly type: ErrorType = ErrorType.UNKNOWN_ERROR,
    public readonly severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    public readonly code: string = 'LENS_ERROR',
    public readonly context: Record<string, any> = {},
    public readonly cause?: Error,
    public readonly retryable: boolean = false
  ) {
    super(message);
    this.name = 'LensError';
    
    if (cause) {
      this.stack = `${this.stack}\nCaused by: ${cause.stack}`;
    }
  }

  toJSON() {
    return {
      name: this.name,
      message: this.message,
      type: this.type,
      severity: this.severity,
      code: this.code,
      context: this.context,
      retryable: this.retryable,
      timestamp: new Date().toISOString()
    };
  }
}

// Circuit breaker states
export enum CircuitState {
  CLOSED = 'CLOSED',     // Normal operation
  OPEN = 'OPEN',         // Circuit tripped, rejecting requests
  HALF_OPEN = 'HALF_OPEN' // Testing if service is back up
}

// Circuit breaker configuration
export interface CircuitBreakerConfig {
  name: string;
  failureThreshold: number;      // Number of failures before opening
  successThreshold: number;      // Number of successes to close from half-open
  timeout: number;               // Time to wait before trying half-open (ms)
  monitoringPeriod: number;      // Window for failure counting (ms)
  expectedErrors?: ErrorType[];   // Errors that should trip the circuit
}

// Retry configuration
export interface RetryConfig {
  maxAttempts: number;
  baseDelay: number;           // Base delay in ms
  maxDelay: number;            // Maximum delay in ms
  backoffMultiplier: number;   // Multiplier for exponential backoff
  jitterMax: number;           // Maximum jitter in ms
  retryableErrors: ErrorType[];
}

// Bulkhead configuration
export interface BulkheadConfig {
  name: string;
  maxConcurrentRequests: number;
  maxQueueSize: number;
  timeout: number;
}

// Rate limiter configuration
export interface RateLimiterConfig {
  requestsPerWindow: number;
  windowSize: number; // in ms
  burstSize: number;
  adaptive: boolean; // Enable adaptive throttling
}

/**
 * Circuit breaker implementation
 */
class CircuitBreaker extends EventEmitter {
  private state: CircuitState = CircuitState.CLOSED;
  private failures: number = 0;
  private successes: number = 0;
  private nextAttempt: number = 0;
  private recentFailures: number[] = [];

  constructor(private config: CircuitBreakerConfig) {
    super();
  }

  async execute<T>(operation: () => Promise<T>): Promise<T> {
    if (this.state === CircuitState.OPEN) {
      if (Date.now() < this.nextAttempt) {
        throw new LensError(
          `Circuit breaker ${this.config.name} is OPEN`,
          ErrorType.SERVICE_UNAVAILABLE,
          ErrorSeverity.HIGH,
          'CIRCUIT_BREAKER_OPEN',
          { circuitName: this.config.name, nextAttempt: this.nextAttempt }
        );
      }
      this.state = CircuitState.HALF_OPEN;
      this.successes = 0;
      this.emit('stateChanged', { name: this.config.name, state: this.state });
    }

    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure(error as Error);
      throw error;
    }
  }

  private onSuccess(): void {
    this.failures = 0;
    
    if (this.state === CircuitState.HALF_OPEN) {
      this.successes++;
      if (this.successes >= this.config.successThreshold) {
        this.state = CircuitState.CLOSED;
        this.emit('stateChanged', { name: this.config.name, state: this.state });
      }
    }
  }

  private onFailure(error: Error): void {
    const lensError = error instanceof LensError ? error : 
      new LensError(error.message, ErrorType.UNKNOWN_ERROR, ErrorSeverity.MEDIUM, 'UNKNOWN', {}, error);
    
    // Check if this error should trip the circuit
    if (this.config.expectedErrors && !this.config.expectedErrors.includes(lensError.type)) {
      return;
    }

    const now = Date.now();
    this.recentFailures.push(now);
    
    // Clean old failures outside monitoring window
    this.recentFailures = this.recentFailures.filter(
      time => now - time < this.config.monitoringPeriod
    );
    
    this.failures = this.recentFailures.length;

    if (this.state === CircuitState.HALF_OPEN) {
      this.state = CircuitState.OPEN;
      this.nextAttempt = now + this.config.timeout;
      this.emit('stateChanged', { name: this.config.name, state: this.state });
    } else if (this.state === CircuitState.CLOSED && this.failures >= this.config.failureThreshold) {
      this.state = CircuitState.OPEN;
      this.nextAttempt = now + this.config.timeout;
      this.emit('stateChanged', { name: this.config.name, state: this.state });
    }
  }

  getState(): CircuitState {
    return this.state;
  }

  getMetrics() {
    return {
      state: this.state,
      failures: this.failures,
      successes: this.successes,
      nextAttempt: this.nextAttempt
    };
  }

  reset(): void {
    this.state = CircuitState.CLOSED;
    this.failures = 0;
    this.successes = 0;
    this.recentFailures = [];
    this.emit('stateChanged', { name: this.config.name, state: this.state });
  }
}

/**
 * Bulkhead isolation implementation
 */
class Bulkhead {
  private activeRequests: number = 0;
  private queue: Array<{ resolve: (value: any) => void; reject: (reason: any) => void; operation: () => Promise<any>; timeout: NodeJS.Timeout }> = [];

  constructor(private config: BulkheadConfig) {}

  async execute<T>(operation: () => Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      if (this.activeRequests < this.config.maxConcurrentRequests) {
        this.executeImmediately(operation, resolve, reject);
      } else if (this.queue.length < this.config.maxQueueSize) {
        this.enqueue(operation, resolve, reject);
      } else {
        reject(new LensError(
          `Bulkhead ${this.config.name} is full`,
          ErrorType.RESOURCE_EXHAUSTED,
          ErrorSeverity.HIGH,
          'BULKHEAD_FULL',
          { 
            bulkheadName: this.config.name, 
            activeRequests: this.activeRequests,
            queueSize: this.queue.length
          }
        ));
      }
    });
  }

  private async executeImmediately<T>(
    operation: () => Promise<T>,
    resolve: (value: T) => void,
    reject: (reason: any) => void
  ): Promise<void> {
    this.activeRequests++;
    
    try {
      const result = await operation();
      resolve(result);
    } catch (error) {
      reject(error);
    } finally {
      this.activeRequests--;
      this.processQueue();
    }
  }

  private enqueue<T>(operation: () => Promise<T>, resolve: (value: T) => void, reject: (reason: any) => void): void {
    const timeout = setTimeout(() => {
      const index = this.queue.findIndex(item => item.resolve === resolve);
      if (index !== -1) {
        this.queue.splice(index, 1);
        reject(new LensError(
          `Bulkhead ${this.config.name} queue timeout`,
          ErrorType.TIMEOUT_ERROR,
          ErrorSeverity.MEDIUM,
          'BULKHEAD_TIMEOUT',
          { bulkheadName: this.config.name }
        ));
      }
    }, this.config.timeout);

    this.queue.push({ resolve, reject, operation, timeout });
  }

  private processQueue(): void {
    if (this.queue.length > 0 && this.activeRequests < this.config.maxConcurrentRequests) {
      const { resolve, reject, operation, timeout } = this.queue.shift()!;
      clearTimeout(timeout);
      this.executeImmediately(operation, resolve, reject);
    }
  }

  getMetrics() {
    return {
      activeRequests: this.activeRequests,
      queueSize: this.queue.length,
      utilization: this.activeRequests / this.config.maxConcurrentRequests
    };
  }
}

/**
 * Rate limiter implementation
 */
class RateLimiter {
  private requests: number[] = [];
  private currentBurst: number = 0;

  constructor(private config: RateLimiterConfig) {}

  async checkLimit(): Promise<boolean> {
    const now = Date.now();
    
    // Clean old requests outside window
    this.requests = this.requests.filter(time => now - time < this.config.windowSize);
    
    // Check rate limit
    if (this.requests.length >= this.config.requestsPerWindow) {
      if (this.currentBurst < this.config.burstSize) {
        this.currentBurst++;
        this.requests.push(now);
        return true;
      }
      return false;
    }
    
    this.requests.push(now);
    
    // Reset burst if we're below rate limit
    if (this.requests.length < this.config.requestsPerWindow * 0.8) {
      this.currentBurst = 0;
    }
    
    return true;
  }

  getMetrics() {
    const now = Date.now();
    const recentRequests = this.requests.filter(time => now - time < this.config.windowSize);
    
    return {
      currentRate: recentRequests.length,
      maxRate: this.config.requestsPerWindow,
      utilization: recentRequests.length / this.config.requestsPerWindow,
      currentBurst: this.currentBurst
    };
  }
}

/**
 * Main resilience manager
 */
export class ResilienceManager extends EventEmitter {
  private readonly tracer = opentelemetry.trace.getTracer('lens-resilience-manager');
  private static instance: ResilienceManager | null = null;
  
  // Resilience components
  private circuitBreakers = new Map<string, CircuitBreaker>();
  private bulkheads = new Map<string, Bulkhead>();
  private rateLimiters = new Map<string, RateLimiter>();
  
  // Fallback cache for graceful degradation
  private fallbackCache = new Map<string, { data: any; timestamp: number; ttl: number }>();
  
  // Error statistics
  private errorStats = new Map<ErrorType, { count: number; lastSeen: number }>();

  private constructor() {
    super();
    this.initializeDefaultComponents();
    this.startCleanupTasks();
  }

  static getInstance(): ResilienceManager {
    if (!ResilienceManager.instance) {
      ResilienceManager.instance = new ResilienceManager();
    }
    return ResilienceManager.instance;
  }

  /**
   * Execute an operation with full resilience patterns
   */
  async executeWithResilience<T>(
    operationName: string,
    operation: () => Promise<T>,
    options: {
      circuitBreaker?: string;
      bulkhead?: string;
      rateLimit?: string;
      retry?: RetryConfig;
      timeout?: number;
      fallback?: () => Promise<T>;
      fallbackCacheKey?: string;
      fallbackCacheTtl?: number;
    } = {}
  ): Promise<OperationResult<T>> {
    return await this.tracer.startActiveSpan('execute-with-resilience', async (span) => {
      const startTime = performance.now();
      let retryCount = 0;
      let lastError: Error | null = null;
      let fallbackUsed = false;

      span.setAttributes({
        'lens.resilience.operation': operationName,
        'lens.resilience.circuit_breaker': options.circuitBreaker || 'none',
        'lens.resilience.bulkhead': options.bulkhead || 'none'
      });

      try {
        // Rate limiting check
        if (options.rateLimit) {
          const rateLimiter = this.rateLimiters.get(options.rateLimit);
          if (rateLimiter && !(await rateLimiter.checkLimit())) {
            throw new LensError(
              `Rate limit exceeded for ${operationName}`,
              ErrorType.RESOURCE_EXHAUSTED,
              ErrorSeverity.MEDIUM,
              'RATE_LIMIT_EXCEEDED',
              { operation: operationName, rateLimiter: options.rateLimit }
            );
          }
        }

        // Execute with retry logic
        const result: T = await this.executeWithRetry(
          () => this.executeWithProtection(operation, options),
          options.retry || this.getDefaultRetryConfig(),
          (attempt, error) => {
            retryCount = attempt;
            lastError = error;
            span.addEvent('retry_attempt', {
              attempt,
              error: error.message,
              error_type: error instanceof LensError ? error.type : 'unknown'
            });
          }
        );

        const duration = performance.now() - startTime;

        // Record success metrics
        performanceMonitor.recordMetric(`${operationName}_success`, 1, 'count');
        performanceMonitor.recordMetric(`${operationName}_duration`, duration, 'ms');

        span.setAttributes({
          'lens.resilience.success': true,
          'lens.resilience.retry_count': retryCount,
          'lens.resilience.duration_ms': duration
        });

        return {
          success: true,
          data: result as T,
          retryCount,
          duration,
          fallbackUsed
        } as OperationResult<T>;

      } catch (error) {
        const lensError = this.classifyError(error as Error, operationName);
        this.recordError(lensError);

        // Attempt fallback
        if (options.fallback) {
          try {
            const fallbackResult = await options.fallback();
            fallbackUsed = true;
            
            // Cache fallback result if requested
            if (options.fallbackCacheKey) {
              this.cacheFallbackResult(
                options.fallbackCacheKey, 
                fallbackResult,
                options.fallbackCacheTtl || 300000 // 5 minutes default
              );
            }

            const duration = performance.now() - startTime;

            span.setAttributes({
              'lens.resilience.success': true,
              'lens.resilience.fallback_used': true,
              'lens.resilience.duration_ms': duration
            });

            return {
              success: true,
              data: fallbackResult as T,
              retryCount,
              duration,
              fallbackUsed
            } as OperationResult<T>;

          } catch (fallbackError) {
            // Fallback failed, try cached fallback
            if (options.fallbackCacheKey) {
              const cached = this.getCachedFallback(options.fallbackCacheKey);
              if (cached) {
                fallbackUsed = true;
                
                const duration = performance.now() - startTime;

                span.setAttributes({
                  'lens.resilience.success': true,
                  'lens.resilience.cached_fallback_used': true,
                  'lens.resilience.duration_ms': duration
                });

                return {
                  success: true,
                  data: cached as T,
                  retryCount,
                  duration,
                  fallbackUsed
                } as OperationResult<T>;
              }
            }
          }
        }

        // No fallback available or fallback failed
        const duration = performance.now() - startTime;

        span.recordException(lensError);
        span.setAttributes({
          'lens.resilience.success': false,
          'lens.resilience.error_type': lensError.type,
          'lens.resilience.error_severity': lensError.severity,
          'lens.resilience.duration_ms': duration
        });

        return {
          success: false,
          error: lensError,
          retryCount,
          duration,
          fallbackUsed
        } as OperationResult<T>;

      } finally {
        span.end();
      }
    });
  }

  /**
   * Register circuit breaker
   */
  registerCircuitBreaker(config: CircuitBreakerConfig): void {
    const circuitBreaker = new CircuitBreaker(config);
    circuitBreaker.on('stateChanged', (event) => {
      this.emit('circuitBreakerStateChanged', event);
      performanceMonitor.recordMetric(
        `circuit_breaker_${config.name}_state_changes`, 
        1, 
        'count',
        { state: event.state }
      );
    });
    
    this.circuitBreakers.set(config.name, circuitBreaker);
  }

  /**
   * Register bulkhead
   */
  registerBulkhead(config: BulkheadConfig): void {
    this.bulkheads.set(config.name, new Bulkhead(config));
  }

  /**
   * Register rate limiter
   */
  registerRateLimiter(name: string, config: RateLimiterConfig): void {
    this.rateLimiters.set(name, new RateLimiter(config));
  }

  /**
   * Get resilience metrics
   */
  getMetrics() {
    const circuitBreakerMetrics = Array.from(this.circuitBreakers.entries()).map(([name, cb]) => ({
      name,
      ...cb.getMetrics()
    }));

    const bulkheadMetrics = Array.from(this.bulkheads.entries()).map(([name, bh]) => ({
      name,
      ...bh.getMetrics()
    }));

    const rateLimiterMetrics = Array.from(this.rateLimiters.entries()).map(([name, rl]) => ({
      name,
      ...rl.getMetrics()
    }));

    const errorStatistics = Array.from(this.errorStats.entries()).map(([type, stats]) => ({
      type,
      ...stats
    }));

    return {
      circuitBreakers: circuitBreakerMetrics,
      bulkheads: bulkheadMetrics,
      rateLimiters: rateLimiterMetrics,
      errors: errorStatistics,
      fallbackCacheSize: this.fallbackCache.size
    };
  }

  /**
   * Private helper methods
   */

  private async executeWithProtection<T>(
    operation: () => Promise<T>,
    options: {
      circuitBreaker?: string;
      bulkhead?: string;
      timeout?: number;
    }
  ): Promise<T> {
    let protectedOperation = operation;

    // Apply timeout
    if (options.timeout) {
      protectedOperation = () => this.withTimeout(operation(), options.timeout!);
    }

    // Apply circuit breaker
    if (options.circuitBreaker) {
      const circuitBreaker = this.circuitBreakers.get(options.circuitBreaker);
      if (circuitBreaker) {
        const cbOperation = protectedOperation;
        protectedOperation = () => circuitBreaker.execute(cbOperation);
      }
    }

    // Apply bulkhead
    if (options.bulkhead) {
      const bulkhead = this.bulkheads.get(options.bulkhead);
      if (bulkhead) {
        const bhOperation = protectedOperation;
        protectedOperation = () => bulkhead.execute(bhOperation);
      }
    }

    return await protectedOperation();
  }

  private async executeWithRetry<T>(
    operation: () => Promise<T>,
    config: RetryConfig,
    onRetry?: (attempt: number, error: Error) => void
  ): Promise<T> {
    let lastError: Error;
    
    for (let attempt = 1; attempt <= config.maxAttempts; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error as Error;
        const lensError = error instanceof LensError ? error : this.classifyError(lastError);
        
        // Check if error is retryable
        if (!config.retryableErrors.includes(lensError.type) || attempt === config.maxAttempts) {
          throw lastError;
        }
        
        if (onRetry) {
          onRetry(attempt, lastError);
        }
        
        // Calculate delay with exponential backoff and jitter
        const baseDelay = config.baseDelay * Math.pow(config.backoffMultiplier, attempt - 1);
        const cappedDelay = Math.min(baseDelay, config.maxDelay);
        const jitter = Math.random() * config.jitterMax;
        const delay = cappedDelay + jitter;
        
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    throw lastError!;
  }

  private withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new LensError(
          `Operation timed out after ${timeoutMs}ms`,
          ErrorType.TIMEOUT_ERROR,
          ErrorSeverity.MEDIUM,
          'TIMEOUT',
          { timeoutMs }
        ));
      }, timeoutMs);

      promise
        .then(resolve)
        .catch(reject)
        .finally(() => clearTimeout(timeoutId));
    });
  }

  private classifyError(error: Error, context?: string): LensError {
    if (error instanceof LensError) {
      return error;
    }

    // Pattern matching for error classification
    const message = error.message.toLowerCase();
    let type = ErrorType.UNKNOWN_ERROR;
    let severity = ErrorSeverity.MEDIUM;
    let retryable = false;

    if (message.includes('timeout') || message.includes('etimedout')) {
      type = ErrorType.TIMEOUT_ERROR;
      retryable = true;
    } else if (message.includes('econnrefused') || message.includes('enotfound')) {
      type = ErrorType.NETWORK_ERROR;
      retryable = true;
    } else if (message.includes('unauthorized') || message.includes('401')) {
      type = ErrorType.AUTHENTICATION_ERROR;
      severity = ErrorSeverity.HIGH;
    } else if (message.includes('forbidden') || message.includes('403')) {
      type = ErrorType.AUTHORIZATION_ERROR;
      severity = ErrorSeverity.HIGH;
    } else if (message.includes('service unavailable') || message.includes('503')) {
      type = ErrorType.SERVICE_UNAVAILABLE;
      severity = ErrorSeverity.HIGH;
      retryable = true;
    } else if (message.includes('validation') || message.includes('invalid')) {
      type = ErrorType.VALIDATION_ERROR;
    }

    return new LensError(
      error.message,
      type,
      severity,
      'CLASSIFIED_ERROR',
      { context, originalError: error.constructor.name },
      error,
      retryable
    );
  }

  private recordError(error: LensError): void {
    const stats = this.errorStats.get(error.type) || { count: 0, lastSeen: 0 };
    stats.count++;
    stats.lastSeen = Date.now();
    this.errorStats.set(error.type, stats);

    // Record metrics
    performanceMonitor.recordMetric(`error_${error.type}`, 1, 'count', {
      severity: error.severity,
      retryable: error.retryable.toString()
    });

    // Emit error event
    this.emit('error', error);
  }

  private cacheFallbackResult<T>(key: string, data: T, ttl: number): void {
    this.fallbackCache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    });
  }

  private getCachedFallback<T>(key: string): T | null {
    const cached = this.fallbackCache.get(key);
    if (!cached) return null;
    
    if (Date.now() - cached.timestamp > cached.ttl) {
      this.fallbackCache.delete(key);
      return null;
    }
    
    return cached.data as T;
  }

  private getDefaultRetryConfig(): RetryConfig {
    return {
      maxAttempts: 3,
      baseDelay: 100,
      maxDelay: 5000,
      backoffMultiplier: 2,
      jitterMax: 100,
      retryableErrors: [
        ErrorType.NETWORK_ERROR,
        ErrorType.TIMEOUT_ERROR,
        ErrorType.SERVICE_UNAVAILABLE
      ]
    };
  }

  private initializeDefaultComponents(): void {
    // Default circuit breakers
    this.registerCircuitBreaker({
      name: 'search-engine',
      failureThreshold: 5,
      successThreshold: 3,
      timeout: 30000,
      monitoringPeriod: 60000,
      expectedErrors: [ErrorType.SERVICE_UNAVAILABLE, ErrorType.TIMEOUT_ERROR]
    });

    this.registerCircuitBreaker({
      name: 'external-api',
      failureThreshold: 3,
      successThreshold: 2,
      timeout: 60000,
      monitoringPeriod: 120000,
      expectedErrors: [ErrorType.NETWORK_ERROR, ErrorType.SERVICE_UNAVAILABLE]
    });

    // Default bulkheads
    this.registerBulkhead({
      name: 'search-requests',
      maxConcurrentRequests: 100,
      maxQueueSize: 500,
      timeout: 10000
    });

    this.registerBulkhead({
      name: 'index-operations',
      maxConcurrentRequests: 10,
      maxQueueSize: 50,
      timeout: 30000
    });

    // Default rate limiters
    this.registerRateLimiter('api', {
      requestsPerWindow: 1000,
      windowSize: 60000, // 1 minute
      burstSize: 100,
      adaptive: true
    });
  }

  private startCleanupTasks(): void {
    // Cleanup expired fallback cache entries every 5 minutes
    setInterval(() => {
      const now = Date.now();
      for (const [key, cached] of this.fallbackCache.entries()) {
        if (now - cached.timestamp > cached.ttl) {
          this.fallbackCache.delete(key);
        }
      }
    }, 300000);

    // Reset old error statistics every hour
    setInterval(() => {
      const now = Date.now();
      const oneHour = 60 * 60 * 1000;
      
      for (const [type, stats] of this.errorStats.entries()) {
        if (now - stats.lastSeen > oneHour) {
          this.errorStats.delete(type);
        }
      }
    }, 3600000);
  }
}

// Export singleton instance and types
export const resilienceManager = ResilienceManager.getInstance();