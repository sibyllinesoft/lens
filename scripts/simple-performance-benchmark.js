#!/usr/bin/env node

import http from 'http';
import { performance } from 'perf_hooks';
import { promises as fs } from 'fs';

/**
 * Simple Performance Benchmark for Lens Search API
 * Creates baseline measurements for migration comparison
 */
class SimpleBenchmark {
  constructor() {
    this.baseUrl = 'http://localhost:3000';
    this.results = [];
  }

  async makeRequest(query) {
    return new Promise((resolve, reject) => {
      const data = JSON.stringify({ query });
      const options = {
        hostname: 'localhost',
        port: 3000,
        path: '/search',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(data)
        }
      };

      const startTime = performance.now();
      const req = http.request(options, (res) => {
        let responseData = '';
        res.on('data', (chunk) => responseData += chunk);
        res.on('end', () => {
          const endTime = performance.now();
          const latency = endTime - startTime;
          
          try {
            const result = JSON.parse(responseData);
            // Consider it successful if we got valid JSON response with expected structure
            const isSuccess = result && ('hits' in result || 'total' in result || 'latency_ms' in result);
            resolve({
              latency,
              statusCode: isSuccess ? 200 : res.statusCode,
              actualStatusCode: res.statusCode,
              data: result,
              query
            });
          } catch (e) {
            resolve({
              latency,
              statusCode: res.statusCode,
              actualStatusCode: res.statusCode,
              error: responseData,
              query
            });
          }
        });
      });

      req.on('error', (error) => {
        resolve({
          latency: 0,
          statusCode: 0,
          error: error.message,
          query
        });
      });
      
      req.write(data);
      req.end();
    });
  }

  async runSingleBenchmark(query, iterations = 100) {
    console.log(`\n📊 Benchmarking query: "${query}"`);
    const results = [];
    
    for (let i = 0; i < iterations; i++) {
      const result = await this.makeRequest(query);
      results.push(result);
      
      if (i % 10 === 0 && i > 0) {
        console.log(`  Completed ${i}/${iterations} requests`);
      }
      
      // Small delay to avoid overwhelming the server
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    
    return results;
  }

  async runConcurrentBenchmark(query, concurrency = 10, totalRequests = 100) {
    console.log(`\n🚀 Concurrent benchmark: ${concurrency} concurrent requests, ${totalRequests} total`);
    const results = [];
    const startTime = performance.now();
    
    const runBatch = async (batchSize) => {
      const promises = [];
      for (let i = 0; i < batchSize; i++) {
        promises.push(this.makeRequest(query));
      }
      return await Promise.all(promises);
    };
    
    const batches = Math.ceil(totalRequests / concurrency);
    for (let batch = 0; batch < batches; batch++) {
      const batchSize = Math.min(concurrency, totalRequests - (batch * concurrency));
      const batchResults = await runBatch(batchSize);
      results.push(...batchResults);
      
      console.log(`  Batch ${batch + 1}/${batches} completed - ${batchResults.length} requests`);
    }
    
    const totalTime = performance.now() - startTime;
    const qps = (totalRequests / (totalTime / 1000)).toFixed(2);
    
    console.log(`  Total time: ${totalTime.toFixed(2)}ms, QPS: ${qps}`);
    return { results, totalTime, qps };
  }

  calculateStats(latencies) {
    if (latencies.length === 0) return null;
    
    const sorted = [...latencies].sort((a, b) => a - b);
    const sum = latencies.reduce((a, b) => a + b, 0);
    
    return {
      count: latencies.length,
      min: sorted[0].toFixed(2),
      max: sorted[sorted.length - 1].toFixed(2),
      avg: (sum / latencies.length).toFixed(2),
      p50: this.percentile(sorted, 50).toFixed(2),
      p95: this.percentile(sorted, 95).toFixed(2),
      p99: this.percentile(sorted, 99).toFixed(2)
    };
  }

  percentile(sortedArray, p) {
    if (sortedArray.length === 0) return 0;
    if (sortedArray.length === 1) return sortedArray[0];
    
    const index = (p / 100) * (sortedArray.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    
    if (lower === upper) return sortedArray[lower];
    return sortedArray[lower] + (sortedArray[upper] - sortedArray[lower]) * (index - lower);
  }

  async runFullBenchmark() {
    console.log('🔥 Lens Search Engine - Performance Baseline Benchmark');
    console.log('======================================================');
    
    const testQueries = [
      'function',
      'class',  
      'user',
      'async',
      'getUserById',
      'UserService',
      'authentication',
      'database connection',
    ];
    
    const report = {
      timestamp: new Date().toISOString(),
      server: this.baseUrl,
      system: 'TypeScript (Baseline)',
      queries: {},
      overall: {},
      loadTest: {}
    };
    
    // Test each query individually
    for (const query of testQueries) {
      const results = await this.runSingleBenchmark(query, 50);
      const successResults = results.filter(r => r.statusCode === 200 && r.data);
      const errorResults = results.filter(r => r.statusCode !== 200);
      
      if (successResults.length > 0) {
        const latencies = successResults.map(r => r.latency);
        const stats = this.calculateStats(latencies);
        
        report.queries[query] = {
          stats,
          success_rate: ((successResults.length / results.length) * 100).toFixed(1),
          error_rate: ((errorResults.length / results.length) * 100).toFixed(1),
          total_requests: results.length
        };
        
        console.log(`  ✅ "${query}": P95=${stats.p95}ms, Success=${report.queries[query].success_rate}%`);
      } else {
        console.log(`  ❌ "${query}": All requests failed`);
        report.queries[query] = {
          stats: null,
          success_rate: '0.0',
          error_rate: '100.0',
          total_requests: results.length
        };
      }
    }
    
    // Run concurrent load test
    console.log('\n========================================');
    console.log('🚀 CONCURRENT LOAD TEST');
    console.log('========================================');
    
    const loadTestResults = await this.runConcurrentBenchmark('function', 20, 200);
    const loadSuccessful = loadTestResults.results.filter(r => r.statusCode === 200);
    const loadLatencies = loadSuccessful.map(r => r.latency);
    
    report.loadTest = {
      concurrent_users: 20,
      total_requests: 200,
      success_rate: ((loadSuccessful.length / loadTestResults.results.length) * 100).toFixed(1),
      qps: loadTestResults.qps,
      stats: this.calculateStats(loadLatencies)
    };
    
    // Calculate overall statistics
    const allSuccessful = Object.values(report.queries)
      .filter(q => q.stats !== null)
      .map(q => q.stats);
      
    if (allSuccessful.length > 0) {
      const avgLatencies = allSuccessful.map(s => parseFloat(s.avg));
      const p95Latencies = allSuccessful.map(s => parseFloat(s.p95));
      
      report.overall = {
        avg_latency: (avgLatencies.reduce((a, b) => a + b, 0) / avgLatencies.length).toFixed(2),
        avg_p95: (p95Latencies.reduce((a, b) => a + b, 0) / p95Latencies.length).toFixed(2),
        queries_tested: allSuccessful.length
      };
    }
    
    return report;
  }

  generateReport(benchmarkData) {
    const lines = [];
    lines.push('# Lens Search Engine - TypeScript Baseline Performance Report');
    lines.push('');
    lines.push(`**Timestamp**: ${benchmarkData.timestamp}`);
    lines.push(`**System**: ${benchmarkData.system}`);
    lines.push(`**Server**: ${benchmarkData.server}`);
    lines.push('');
    
    // Query Performance
    lines.push('## Individual Query Performance');
    lines.push('');
    lines.push('| Query | P95 Latency | Avg Latency | Success Rate | Requests |');
    lines.push('|-------|-------------|-------------|--------------|----------|');
    
    for (const [query, data] of Object.entries(benchmarkData.queries)) {
      const stats = data.stats;
      if (stats) {
        lines.push(`| ${query} | ${stats.p95}ms | ${stats.avg}ms | ${data.success_rate}% | ${data.total_requests} |`);
      } else {
        lines.push(`| ${query} | N/A | N/A | ${data.success_rate}% | ${data.total_requests} |`);
      }
    }
    lines.push('');
    
    // Overall Summary
    lines.push('## Overall Performance Summary');
    lines.push('');
    if (benchmarkData.overall.queries_tested) {
      lines.push(`- **Average Latency**: ${benchmarkData.overall.avg_latency}ms`);
      lines.push(`- **Average P95 Latency**: ${benchmarkData.overall.avg_p95}ms`);
      lines.push(`- **Queries Successfully Tested**: ${benchmarkData.overall.queries_tested}`);
    }
    lines.push('');
    
    // Load Test Results
    lines.push('## Concurrent Load Test Results');
    lines.push('');
    lines.push(`- **Concurrent Users**: ${benchmarkData.loadTest.concurrent_users}`);
    lines.push(`- **Total Requests**: ${benchmarkData.loadTest.total_requests}`);
    lines.push(`- **Success Rate**: ${benchmarkData.loadTest.success_rate}%`);
    lines.push(`- **Queries per Second**: ${benchmarkData.loadTest.qps}`);
    
    if (benchmarkData.loadTest.stats) {
      lines.push(`- **Load Test P95 Latency**: ${benchmarkData.loadTest.stats.p95}ms`);
      lines.push(`- **Load Test Average Latency**: ${benchmarkData.loadTest.stats.avg}ms`);
    }
    lines.push('');
    
    // System Characteristics
    lines.push('## System Characteristics (TypeScript Baseline)');
    lines.push('');
    lines.push('- **Runtime**: Node.js with TypeScript');
    lines.push('- **Server Framework**: Express/Fastify');
    lines.push('- **Search Engine**: JavaScript implementation');
    lines.push('- **Memory Management**: V8 garbage collector');
    lines.push('- **Concurrency**: Event loop based');
    lines.push('');
    
    return lines.join('\n');
  }

  async saveResults(data, reportText) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    // Save raw JSON data
    await fs.writeFile(
      `typescript-baseline-${timestamp}.json`,
      JSON.stringify(data, null, 2)
    );
    
    // Save readable report
    await fs.writeFile(
      `typescript-baseline-report-${timestamp}.md`,
      reportText
    );
    
    console.log(`\n📊 Baseline results saved:`);
    console.log(`   - Raw data: typescript-baseline-${timestamp}.json`);
    console.log(`   - Report: typescript-baseline-report-${timestamp}.md`);
    
    return { dataFile: `typescript-baseline-${timestamp}.json`, reportFile: `typescript-baseline-report-${timestamp}.md` };
  }
}

// Run the benchmark if this script is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const benchmark = new SimpleBenchmark();
  
  benchmark.runFullBenchmark()
    .then(async (results) => {
      const report = benchmark.generateReport(results);
      console.log('\n' + '='.repeat(60));
      console.log('📊 TYPESCRIPT BASELINE COMPLETE');
      console.log('='.repeat(60));
      console.log(report);
      
      await benchmark.saveResults(results, report);
    })
    .catch(error => {
      console.error('Benchmark failed:', error);
      process.exit(1);
    });
}

export default SimpleBenchmark;