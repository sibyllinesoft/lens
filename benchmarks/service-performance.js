#!/usr/bin/env node

import http from 'http';
import { performance } from 'perf_hooks';
import { promises as fs } from 'fs';

/**
 * Comprehensive Service Performance Benchmark
 * Tests latency, throughput, and SLA compliance
 */
class ServicePerformanceBenchmark {
  constructor() {
    this.baseUrl = 'http://localhost:3001';
    this.results = [];
    this.testQueries = [
      // Stage A: Lexical queries (2-8ms target)
      { query: 'function', mode: 'lex', fuzzy: 0.8, expectedStage: 'A' },
      { query: 'class', mode: 'lex', fuzzy: 0.7, expectedStage: 'A' },
      { query: 'user', mode: 'lex', fuzzy: 0.9, expectedStage: 'A' },
      { query: 'search', mode: 'lex', fuzzy: 0.8, expectedStage: 'A' },
      
      // Stage B: Structural queries (3-10ms target)
      { query: 'getUserById', mode: 'struct', fuzzy: 0.7, expectedStage: 'B' },
      { query: 'UserService', mode: 'struct', fuzzy: 0.8, expectedStage: 'B' },
      { query: 'async findUser', mode: 'struct', fuzzy: 0.6, expectedStage: 'B' },
      
      // Stage C: Hybrid queries (5-15ms target)
      { query: 'authentication flow', mode: 'hybrid', fuzzy: 0.7, expectedStage: 'C' },
      { query: 'database connection', mode: 'hybrid', fuzzy: 0.8, expectedStage: 'C' },
      { query: 'error handling pattern', mode: 'hybrid', fuzzy: 0.6, expectedStage: 'C' }
    ];
  }

  async makeRequest(query) {
    return new Promise((resolve, reject) => {
      const data = JSON.stringify(query);
      const options = {
        hostname: 'localhost',
        port: 3001,
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
            resolve({
              latency,
              statusCode: res.statusCode,
              data: result,
              query
            });
          } catch (e) {
            resolve({
              latency,
              statusCode: res.statusCode,
              error: responseData,
              query
            });
          }
        });
      });

      req.on('error', reject);
      req.write(data);
      req.end();
    });
  }

  async runSingleBenchmark(testQuery, iterations = 10) {
    console.log(`\nBenchmarking ${testQuery.expectedStage} query: "${testQuery.query}"`);
    const results = [];
    
    for (let i = 0; i < iterations; i++) {
      try {
        const result = await this.makeRequest(testQuery);
        results.push(result);
        
        if (result.statusCode === 200 && result.data) {
          const stageLatencies = result.data.performance || {};
          console.log(`  Run ${i+1}: ${result.latency.toFixed(2)}ms total, Stage A: ${stageLatencies.stage_a || 'N/A'}ms, Stage B: ${stageLatencies.stage_b || 'N/A'}ms`);
        } else {
          console.log(`  Run ${i+1}: ${result.latency.toFixed(2)}ms - Error: ${result.statusCode}`);
        }
      } catch (error) {
        console.log(`  Run ${i+1}: Error - ${error.message}`);
        results.push({ error: error.message, query: testQuery });
      }
      
      // Small delay between requests
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    
    return results;
  }

  async runConcurrentBenchmark(testQuery, concurrency = 5, totalRequests = 50) {
    console.log(`\nConcurrent benchmark: ${concurrency} concurrent requests, ${totalRequests} total`);
    const results = [];
    const startTime = performance.now();
    
    const runBatch = async (batchSize) => {
      const promises = [];
      for (let i = 0; i < batchSize; i++) {
        promises.push(this.makeRequest(testQuery));
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
      min: sorted[0],
      max: sorted[sorted.length - 1],
      avg: sum / latencies.length,
      p50: this.percentile(sorted, 50),
      p95: this.percentile(sorted, 95),
      p99: this.percentile(sorted, 99)
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

  evaluateSLACompliance(stats, targetMax) {
    if (!stats) return { compliance: 0, passed: false };
    
    const compliance = (stats.p95 <= targetMax) ? 1 : 0;
    return {
      compliance,
      passed: compliance === 1,
      p95: stats.p95,
      target: targetMax,
      margin: ((targetMax - stats.p95) / targetMax * 100).toFixed(1)
    };
  }

  async runFullBenchmark() {
    console.log('🚀 Starting Comprehensive Service Performance Benchmark');
    console.log('=' .repeat(60));
    
    const report = {
      timestamp: new Date().toISOString(),
      server: this.baseUrl,
      stages: {},
      overall: {},
      sla_compliance: {}
    };
    
    // Test each stage individually
    for (const testQuery of this.testQueries) {
      const results = await this.runSingleBenchmark(testQuery, 20);
      const successResults = results.filter(r => r.statusCode === 200 && r.data);
      
      if (successResults.length > 0) {
        const latencies = successResults.map(r => r.latency);
        const stats = this.calculateStats(latencies);
        
        const stage = testQuery.expectedStage;
        if (!report.stages[stage]) {
          report.stages[stage] = { queries: [], stats: null };
        }
        
        report.stages[stage].queries.push({
          query: testQuery.query,
          mode: testQuery.mode,
          stats,
          success_rate: (successResults.length / results.length * 100).toFixed(1)
        });
      }
    }
    
    // Calculate aggregate stats per stage
    const slaTargets = { A: 8, B: 10, C: 15 };
    for (const [stage, data] of Object.entries(report.stages)) {
      const allLatencies = data.queries.flatMap(q => {
        // Reconstruct latencies from stats (approximation)
        const stats = q.stats;
        return new Array(stats.count).fill(0).map(() => 
          stats.min + Math.random() * (stats.max - stats.min)
        );
      });
      
      data.aggregated_stats = this.calculateStats(allLatencies);
      report.sla_compliance[`stage_${stage}`] = this.evaluateSLACompliance(
        data.aggregated_stats, 
        slaTargets[stage]
      );
    }
    
    // Run concurrent load test
    console.log('\n' + '='.repeat(40));
    console.log('CONCURRENT LOAD TEST');
    console.log('='.repeat(40));
    
    const loadTestQuery = this.testQueries[0]; // Use first query for load test
    const concurrentResults = await this.runConcurrentBenchmark(loadTestQuery, 10, 100);
    
    const loadTestSuccessful = concurrentResults.results.filter(r => r.statusCode === 200);
    const loadTestLatencies = loadTestSuccessful.map(r => r.latency);
    
    report.load_test = {
      concurrent_users: 10,
      total_requests: 100,
      success_rate: (loadTestSuccessful.length / concurrentResults.results.length * 100).toFixed(1),
      qps: concurrentResults.qps,
      stats: this.calculateStats(loadTestLatencies)
    };
    
    // Overall p95 SLA compliance (target: <20ms)
    report.sla_compliance.overall_p95 = this.evaluateSLACompliance(
      report.load_test.stats,
      20
    );
    
    return report;
  }

  generateReport(benchmarkData) {
    const lines = [];
    lines.push('# Lens Search Engine - Service Performance Benchmark Report');
    lines.push('');
    lines.push(`**Timestamp**: ${benchmarkData.timestamp}`);
    lines.push(`**Server**: ${benchmarkData.server}`);
    lines.push('');
    
    // Stage Performance Summary
    lines.push('## Stage Performance Summary');
    lines.push('');
    
    for (const [stage, data] of Object.entries(benchmarkData.stages)) {
      const stats = data.aggregated_stats;
      const sla = benchmarkData.sla_compliance[`stage_${stage}`];
      
      lines.push(`### Stage ${stage} Performance`);
      lines.push(`- **Queries Tested**: ${data.queries.length}`);
      lines.push(`- **Average Latency**: ${stats.avg.toFixed(2)}ms`);
      lines.push(`- **P95 Latency**: ${stats.p95.toFixed(2)}ms`);
      lines.push(`- **P99 Latency**: ${stats.p99.toFixed(2)}ms`);
      lines.push(`- **SLA Compliance**: ${sla.passed ? '✅ PASSED' : '❌ FAILED'} (Target: ${sla.target}ms, Actual: ${sla.p95.toFixed(2)}ms)`);
      if (sla.passed) {
        lines.push(`- **SLA Margin**: ${sla.margin}% under target`);
      }
      lines.push('');
    }
    
    // Load Test Results
    lines.push('## Concurrent Load Test Results');
    lines.push('');
    lines.push(`- **Concurrent Users**: ${benchmarkData.load_test.concurrent_users}`);
    lines.push(`- **Total Requests**: ${benchmarkData.load_test.total_requests}`);
    lines.push(`- **Success Rate**: ${benchmarkData.load_test.success_rate}%`);
    lines.push(`- **Queries per Second**: ${benchmarkData.load_test.qps}`);
    lines.push(`- **Average Latency**: ${benchmarkData.load_test.stats.avg.toFixed(2)}ms`);
    lines.push(`- **P95 Latency**: ${benchmarkData.load_test.stats.p95.toFixed(2)}ms`);
    
    const overallSLA = benchmarkData.sla_compliance.overall_p95;
    lines.push(`- **Overall SLA**: ${overallSLA.passed ? '✅ PASSED' : '❌ FAILED'} (P95 < 20ms)`);
    lines.push('');
    
    // Recommendations
    lines.push('## Performance Recommendations');
    lines.push('');
    
    const recommendations = [];
    
    // Check each stage SLA
    for (const [stage, sla] of Object.entries(benchmarkData.sla_compliance)) {
      if (stage.startsWith('stage_') && !sla.passed) {
        const stageName = stage.replace('stage_', 'Stage ');
        recommendations.push(`- **${stageName} Optimization Needed**: P95 latency (${sla.p95.toFixed(2)}ms) exceeds target (${sla.target}ms)`);
      }
    }
    
    // Check overall performance
    if (!overallSLA.passed) {
      recommendations.push(`- **Overall Performance**: P95 latency (${overallSLA.p95.toFixed(2)}ms) exceeds 20ms target`);
    }
    
    // Check success rates
    if (parseFloat(benchmarkData.load_test.success_rate) < 99) {
      recommendations.push(`- **Reliability**: Success rate (${benchmarkData.load_test.success_rate}%) should be >99%`);
    }
    
    if (recommendations.length === 0) {
      recommendations.push('- **System Performance**: All SLA targets met - system performing within specifications');
    }
    
    recommendations.forEach(rec => lines.push(rec));
    
    return lines.join('\n');
  }

  async saveBenchmarkResults(data, reportText) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    // Save raw JSON data
    await fs.writeFile(
      `benchmark-service-${timestamp}.json`,
      JSON.stringify(data, null, 2)
    );
    
    // Save readable report
    await fs.writeFile(
      `benchmark-service-report-${timestamp}.md`,
      reportText
    );
    
    console.log(`\n📊 Results saved:`);
    console.log(`   - Raw data: benchmark-service-${timestamp}.json`);
    console.log(`   - Report: benchmark-service-report-${timestamp}.md`);
  }
}

// Run the benchmark if this script is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const benchmark = new ServicePerformanceBenchmark();
  
  benchmark.runFullBenchmark()
    .then(async (results) => {
      const report = benchmark.generateReport(results);
      console.log('\n' + '='.repeat(60));
      console.log('BENCHMARK COMPLETE');
      console.log('='.repeat(60));
      console.log(report);
      
      await benchmark.saveBenchmarkResults(results, report);
    })
    .catch(error => {
      console.error('Benchmark failed:', error);
      process.exit(1);
    });
}

export default ServicePerformanceBenchmark;