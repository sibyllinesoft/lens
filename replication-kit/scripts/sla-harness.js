#!/usr/bin/env node

/**
 * Lens v2.2 SLA Harness for External Reproduction
 * Enforces 150ms SLA timing and measurement protocols
 */

import { performance } from 'perf_hooks';
import { EventEmitter } from 'events';

export class SLAHarness extends EventEmitter {
    constructor(options = {}) {
        super();
        this.slaMs = options.slaMs || 150;
        this.concurrentLimit = options.concurrentLimit || 10;
        this.timeoutBuffer = options.timeoutBuffer || 10; // Extra buffer for network/processing
        
        this.stats = {
            totalQueries: 0,
            slaCompliant: 0,
            timedOut: 0,
            errors: 0,
            latencies: []
        };
    }

    /**
     * Execute a query with SLA enforcement
     * @param {Function} queryFn - Function that executes the query
     * @param {Object} query - Query object with id, text, etc.
     * @returns {Object} Result with timing and SLA compliance
     */
    async executeQuery(queryFn, query) {
        const startTime = performance.now();
        let result = null;
        let error = null;
        let timedOut = false;

        try {
            // Create timeout promise
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('SLA_TIMEOUT')), 
                    this.slaMs + this.timeoutBuffer);
            });

            // Race query execution against timeout
            result = await Promise.race([
                queryFn(query),
                timeoutPromise
            ]);

        } catch (err) {
            error = err;
            if (err.message === 'SLA_TIMEOUT') {
                timedOut = true;
            }
        }

        const endTime = performance.now();
        const latencyMs = endTime - startTime;
        const withinSLA = latencyMs <= this.slaMs;

        // Update statistics
        this.stats.totalQueries++;
        if (withinSLA && !error) {
            this.stats.slaCompliant++;
        }
        if (timedOut) {
            this.stats.timedOut++;
        }
        if (error) {
            this.stats.errors++;
        }
        this.stats.latencies.push(latencyMs);

        const queryResult = {
            queryId: query.id,
            query: query.text,
            latencyMs: latencyMs,
            withinSLA: withinSLA,
            timedOut: timedOut,
            error: error?.message,
            result: result,
            timestamp: new Date().toISOString()
        };

        // Emit events for monitoring
        this.emit('query:complete', queryResult);
        if (!withinSLA) {
            this.emit('query:sla-violation', queryResult);
        }
        if (error) {
            this.emit('query:error', queryResult);
        }

        return queryResult;
    }

    /**
     * Execute multiple queries with concurrency control
     */
    async executeBatch(queryFn, queries) {
        console.log(`🔄 Executing ${queries.length} queries with SLA harness`);
        console.log(`⏱️  SLA: ${this.slaMs}ms timeout`);
        console.log(`🔀 Concurrency: ${this.concurrentLimit} parallel queries`);

        const results = [];
        const semaphore = new Semaphore(this.concurrentLimit);

        const executeWithSemaphore = async (query) => {
            await semaphore.acquire();
            try {
                return await this.executeQuery(queryFn, query);
            } finally {
                semaphore.release();
            }
        };

        const promises = queries.map(executeWithSemaphore);
        const batchResults = await Promise.all(promises);

        results.push(...batchResults);

        console.log(`✅ Batch complete: ${this.stats.slaCompliant}/${this.stats.totalQueries} within SLA`);
        
        return results;
    }

    /**
     * Get performance statistics
     */
    getStats() {
        const latencies = this.stats.latencies.sort((a, b) => a - b);
        const n = latencies.length;
        
        return {
            totalQueries: this.stats.totalQueries,
            slaCompliant: this.stats.slaCompliant,
            slaComplianceRate: this.stats.slaCompliant / this.stats.totalQueries,
            timedOut: this.stats.timedOut,
            errors: this.stats.errors,
            latency: {
                p50: latencies[Math.floor(n * 0.5)] || 0,
                p95: latencies[Math.floor(n * 0.95)] || 0,
                p99: latencies[Math.floor(n * 0.99)] || 0,
                mean: latencies.reduce((a, b) => a + b, 0) / n || 0,
                min: latencies[0] || 0,
                max: latencies[n - 1] || 0
            }
        };
    }

    /**
     * Generate SLA compliance report
     */
    generateReport() {
        const stats = this.getStats();
        
        const report = `# SLA Harness Report

## Configuration
- **SLA Threshold:** ${this.slaMs}ms
- **Timeout Buffer:** ${this.timeoutBuffer}ms
- **Concurrency Limit:** ${this.concurrentLimit}

## Results Summary
- **Total Queries:** ${stats.totalQueries}
- **SLA Compliant:** ${stats.slaCompliant} (${(stats.slaComplianceRate * 100).toFixed(1)}%)
- **Timed Out:** ${stats.timedOut}
- **Errors:** ${stats.errors}

## Latency Distribution
- **p50:** ${stats.latency.p50.toFixed(2)}ms
- **p95:** ${stats.latency.p95.toFixed(2)}ms
- **p99:** ${stats.latency.p99.toFixed(2)}ms
- **Mean:** ${stats.latency.mean.toFixed(2)}ms
- **Range:** ${stats.latency.min.toFixed(2)}ms - ${stats.latency.max.toFixed(2)}ms

## Validation
${stats.slaComplianceRate >= 0.95 ? '✅' : '❌'} SLA compliance ≥ 95%
${stats.latency.p99 <= this.slaMs * 1.1 ? '✅' : '❌'} p99 latency within bounds
${stats.errors / stats.totalQueries <= 0.001 ? '✅' : '❌'} Error rate ≤ 0.1%

Generated: ${new Date().toISOString()}
`;

        return report;
    }
}

/**
 * Simple semaphore for concurrency control
 */
class Semaphore {
    constructor(capacity) {
        this.capacity = capacity;
        this.current = 0;
        this.queue = [];
    }

    async acquire() {
        if (this.current >= this.capacity) {
            await new Promise(resolve => this.queue.push(resolve));
        }
        this.current++;
    }

    release() {
        this.current--;
        if (this.queue.length > 0) {
            const next = this.queue.shift();
            next();
        }
    }
}

// Example usage
if (import.meta.url === `file://${process.argv[1]}`) {
    console.log('SLA Harness ready for query execution');
    console.log('Import this module and use SLAHarness class for benchmark reproduction');
}
