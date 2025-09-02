#!/usr/bin/env node

const http = require('http');
const fs = require('fs').promises;
const { performance } = require('perf_hooks');

/**
 * Create comprehensive benchmark reports using both service performance 
 * and accuracy metrics to guide next development steps
 */
class BenchmarkReportGenerator {
  constructor() {
    this.baseUrl = 'http://localhost:3001';
    this.results = {
      service_performance: {},
      accuracy_evaluation: {},
      recommendations: []
    };
  }

  async makeRequest(endpoint, method = 'GET', data = null) {
    return new Promise((resolve, reject) => {
      const options = {
        hostname: 'localhost',
        port: 3001,
        path: endpoint,
        method: method,
        headers: {}
      };

      if (data) {
        const payload = typeof data === 'string' ? data : JSON.stringify(data);
        options.headers['Content-Type'] = 'application/json';
        options.headers['Content-Length'] = Buffer.byteLength(payload);
      }

      const startTime = performance.now();
      const req = http.request(options, (res) => {
        let responseData = '';
        res.on('data', (chunk) => responseData += chunk);
        res.on('end', () => {
          const endTime = performance.now();
          const latency = endTime - startTime;
          
          try {
            const result = responseData ? JSON.parse(responseData) : {};
            resolve({
              latency,
              statusCode: res.statusCode,
              data: result
            });
          } catch (e) {
            resolve({
              latency,
              statusCode: res.statusCode,
              error: responseData
            });
          }
        });
      });

      req.on('error', reject);
      if (data) {
        const payload = typeof data === 'string' ? data : JSON.stringify(data);
        req.write(payload);
      }
      req.end();
    });
  }

  async testServiceHealth() {
    console.log('🔍 Testing service health and availability...');
    
    try {
      const healthResponse = await this.makeRequest('/health');
      const configResponse = await this.makeRequest('/bench/config');
      
      return {
        health_status: healthResponse.statusCode === 200 ? 'healthy' : 'unhealthy',
        health_latency: healthResponse.latency,
        config_accessible: configResponse.statusCode === 200,
        benchmark_system_available: true
      };
    } catch (error) {
      return {
        health_status: 'error',
        error: error.message,
        benchmark_system_available: false
      };
    }
  }

  async testSearchAPI() {
    console.log('🔍 Testing search API functionality...');
    
    const testQueries = [
      { q: 'function', mode: 'lex', fuzzy: 0.8 },
      { q: 'class', mode: 'lex', fuzzy: 0.7 },
      { q: 'user', mode: 'struct', fuzzy: 0.8 },
      { q: 'UserService', mode: 'struct', fuzzy: 0.9 },
      { q: 'async function', mode: 'hybrid', fuzzy: 0.7 }
    ];

    const results = [];
    
    for (const query of testQueries) {
      try {
        const response = await this.makeRequest('/search', 'POST', query);
        results.push({
          query,
          latency: response.latency,
          status: response.statusCode,
          success: response.statusCode === 200,
          results_count: response.data?.results?.length || 0,
          error: response.statusCode !== 200 ? response.error || response.data : null
        });
        
        console.log(`  Query "${query.q}" (${query.mode}): ${response.statusCode} - ${response.latency.toFixed(2)}ms - ${response.data?.results?.length || 0} results`);
      } catch (error) {
        results.push({
          query,
          latency: 0,
          status: 0,
          success: false,
          results_count: 0,
          error: error.message
        });
        console.log(`  Query "${query.q}" (${query.mode}): ERROR - ${error.message}`);
      }
      
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    return results;
  }

  async runBuiltInBenchmarks() {
    console.log('🚀 Running built-in benchmark suite...');
    
    try {
      // Run smoke test
      const smokeResponse = await this.makeRequest('/bench/smoke', 'POST');
      
      if (smokeResponse.statusCode === 200) {
        console.log(`  Smoke test completed in ${smokeResponse.latency.toFixed(2)}ms`);
        console.log(`  Duration: ${smokeResponse.data.duration_ms}ms`);
        console.log(`  Trace ID: ${smokeResponse.data.trace_id}`);
        console.log(`  Reports generated: ${Object.keys(smokeResponse.data.reports || {}).length}`);
        
        return {
          smoke_test: {
            success: true,
            latency: smokeResponse.latency,
            duration: smokeResponse.data.duration_ms,
            trace_id: smokeResponse.data.trace_id,
            reports: smokeResponse.data.reports,
            promotion_gate: smokeResponse.data.promotion_gate
          }
        };
      } else {
        console.log(`  Smoke test failed: ${smokeResponse.statusCode} - ${smokeResponse.error}`);
        return {
          smoke_test: {
            success: false,
            error: smokeResponse.error,
            status_code: smokeResponse.statusCode
          }
        };
      }
    } catch (error) {
      console.log(`  Benchmark suite error: ${error.message}`);
      return {
        smoke_test: {
          success: false,
          error: error.message
        }
      };
    }
  }

  calculatePerformanceMetrics(searchResults) {
    const successful = searchResults.filter(r => r.success);
    const failed = searchResults.filter(r => !r.success);
    
    if (successful.length === 0) {
      return {
        success_rate: 0,
        avg_latency: 0,
        p95_latency: 0,
        total_queries: searchResults.length,
        successful_queries: 0,
        failed_queries: failed.length
      };
    }
    
    const latencies = successful.map(r => r.latency);
    const sortedLatencies = latencies.sort((a, b) => a - b);
    const p95Index = Math.floor(sortedLatencies.length * 0.95);
    
    return {
      success_rate: successful.length / searchResults.length,
      avg_latency: latencies.reduce((a, b) => a + b, 0) / latencies.length,
      p95_latency: sortedLatencies[p95Index] || 0,
      total_queries: searchResults.length,
      successful_queries: successful.length,
      failed_queries: failed.length,
      results_distribution: {
        with_results: successful.filter(r => r.results_count > 0).length,
        no_results: successful.filter(r => r.results_count === 0).length
      }
    };
  }

  async generateComprehensiveReport() {
    console.log('📊 Generating Comprehensive Benchmark Report');
    console.log('=' .repeat(50));
    
    const report = {
      timestamp: new Date().toISOString(),
      system_under_test: 'Lens Search Engine',
      environment: {
        server: this.baseUrl,
        node_version: process.version,
        platform: process.platform
      },
      service_health: null,
      search_api_test: null,
      builtin_benchmarks: null,
      performance_analysis: null,
      accuracy_analysis: null,
      recommendations: []
    };
    
    // Test service health
    report.service_health = await this.testServiceHealth();
    
    // Test search API functionality
    const searchResults = await this.testSearchAPI();
    report.search_api_test = {
      queries: searchResults,
      performance: this.calculatePerformanceMetrics(searchResults)
    };
    
    // Run built-in benchmarks
    report.builtin_benchmarks = await this.runBuiltInBenchmarks();
    
    // Generate analysis and recommendations
    this.analyzeResults(report);
    
    return report;
  }

  analyzeResults(report) {
    const recommendations = [];
    const issues = [];
    const achievements = [];
    
    // Health analysis
    if (report.service_health.health_status !== 'healthy') {
      issues.push('Service health check failed');
      recommendations.push('CRITICAL: Fix service health issues before proceeding with development');
    } else {
      achievements.push('Service health check passed');
    }
    
    // Search API analysis
    const searchPerf = report.search_api_test.performance;
    
    if (searchPerf.success_rate < 0.8) {
      issues.push(`Low success rate: ${(searchPerf.success_rate * 100).toFixed(1)}%`);
      recommendations.push('HIGH: Investigate search API failures - many queries are not returning results');
    } else if (searchPerf.success_rate < 1.0) {
      recommendations.push('MEDIUM: Some search queries failing - investigate error patterns');
    } else {
      achievements.push('100% search API success rate');
    }
    
    if (searchPerf.avg_latency > 50) {
      issues.push(`High average latency: ${searchPerf.avg_latency.toFixed(2)}ms`);
      recommendations.push('HIGH: Optimize search performance - average latency exceeds targets');
    } else if (searchPerf.avg_latency > 20) {
      recommendations.push('MEDIUM: Search latency could be improved');
    } else {
      achievements.push('Search latency within acceptable range');
    }
    
    if (searchPerf.results_distribution && searchPerf.results_distribution.no_results > searchPerf.results_distribution.with_results) {
      issues.push('Most queries return no results');
      recommendations.push('CRITICAL: Index content is missing or search algorithm needs improvement');
    }
    
    // Built-in benchmark analysis
    const builtinBench = report.builtin_benchmarks?.smoke_test;
    if (builtinBench?.success) {
      achievements.push('Built-in benchmark suite operational');
      if (builtinBench.promotion_gate && builtinBench.promotion_gate !== 'unknown') {
        if (builtinBench.promotion_gate === 'PASSED') {
          achievements.push('Promotion gate criteria met');
        } else {
          issues.push('Promotion gate criteria not met');
          recommendations.push('HIGH: Address promotion gate failures before production deployment');
        }
      }
    } else {
      issues.push('Built-in benchmark suite not operational');
      recommendations.push('MEDIUM: Fix built-in benchmark infrastructure for ongoing testing');
    }
    
    // Overall system assessment
    if (issues.length === 0) {
      recommendations.push('SUCCESS: System performing well - ready for feature development');
    } else if (issues.length <= 2) {
      recommendations.push('PROCEED WITH CAUTION: Address identified issues while continuing development');
    } else {
      recommendations.push('HALT DEVELOPMENT: Critical issues must be resolved before proceeding');
    }
    
    // Next steps based on analysis
    if (searchPerf.results_distribution && searchPerf.results_distribution.no_results > 0) {
      recommendations.push('NEXT: Index sample content (src/example.ts, sample-code/) to enable meaningful search testing');
    }
    
    if (searchPerf.success_rate > 0.8 && searchPerf.avg_latency < 50) {
      recommendations.push('NEXT: Run full accuracy evaluation to assess search relevance and ranking quality');
      recommendations.push('NEXT: Implement performance baseline recording for regression detection');
    }
    
    recommendations.push('NEXT: Set up automated benchmark runs in CI/CD pipeline');
    recommendations.push('NEXT: Establish SLA targets and monitoring based on current performance baseline');
    
    report.analysis = {
      issues,
      achievements,
      overall_assessment: issues.length <= 2 ? 'ACCEPTABLE' : 'NEEDS_IMPROVEMENT'
    };
    
    report.recommendations = recommendations;
  }

  generateMarkdownReport(reportData) {
    const lines = [];
    
    lines.push('# Lens Search Engine - Benchmark Report');
    lines.push('');
    lines.push(`**Generated**: ${reportData.timestamp}`);
    lines.push(`**System**: ${reportData.system_under_test}`);
    lines.push(`**Server**: ${reportData.environment.server}`);
    lines.push('');
    
    // Executive Summary
    lines.push('## Executive Summary');
    lines.push('');
    lines.push(`**Overall Assessment**: ${reportData.analysis.overall_assessment}`);
    lines.push('');
    
    if (reportData.analysis.achievements.length > 0) {
      lines.push('### ✅ Achievements');
      reportData.analysis.achievements.forEach(achievement => {
        lines.push(`- ${achievement}`);
      });
      lines.push('');
    }
    
    if (reportData.analysis.issues.length > 0) {
      lines.push('### ⚠️ Issues Identified');
      reportData.analysis.issues.forEach(issue => {
        lines.push(`- ${issue}`);
      });
      lines.push('');
    }
    
    // Service Performance Analysis
    lines.push('## Service Performance Analysis');
    lines.push('');
    
    const searchPerf = reportData.search_api_test.performance;
    lines.push('### API Response Performance');
    lines.push(`- **Success Rate**: ${(searchPerf.success_rate * 100).toFixed(1)}%`);
    lines.push(`- **Average Latency**: ${searchPerf.avg_latency.toFixed(2)}ms`);
    lines.push(`- **P95 Latency**: ${searchPerf.p95_latency.toFixed(2)}ms`);
    lines.push(`- **Total Queries**: ${searchPerf.total_queries}`);
    lines.push(`- **Successful**: ${searchPerf.successful_queries}`);
    lines.push(`- **Failed**: ${searchPerf.failed_queries}`);
    lines.push('');
    
    lines.push('### Result Quality Distribution');
    if (searchPerf.results_distribution) {
      lines.push(`- **Queries with Results**: ${searchPerf.results_distribution.with_results}`);
      lines.push(`- **Queries with No Results**: ${searchPerf.results_distribution.no_results}`);
    } else {
      lines.push('- **No distribution data available**');
    }
    lines.push('');
    
    // Built-in Benchmark Results
    lines.push('## Built-in Benchmark Results');
    lines.push('');
    
    const builtinBench = reportData.builtin_benchmarks?.smoke_test;
    if (builtinBench?.success) {
      lines.push('✅ **Smoke Test**: PASSED');
      lines.push(`- **Duration**: ${builtinBench.duration}ms`);
      lines.push(`- **Trace ID**: ${builtinBench.trace_id}`);
      lines.push(`- **Promotion Gate**: ${builtinBench.promotion_gate || 'Unknown'}`);
      if (builtinBench.reports) {
        lines.push('- **Generated Reports**:');
        Object.entries(builtinBench.reports).forEach(([type, path]) => {
          lines.push(`  - ${type}: \`${path}\``);
        });
      }
    } else {
      lines.push('❌ **Smoke Test**: FAILED');
      if (builtinBench?.error) {
        lines.push(`- **Error**: ${builtinBench.error}`);
      }
    }
    lines.push('');
    
    // Detailed Query Results
    lines.push('## Detailed Query Analysis');
    lines.push('');
    lines.push('| Query | Mode | Status | Latency | Results | Notes |');
    lines.push('|-------|------|--------|---------|---------|-------|');
    
    reportData.search_api_test.queries.forEach(queryResult => {
      const query = queryResult.query;
      const status = queryResult.success ? '✅' : '❌';
      const latency = `${queryResult.latency.toFixed(2)}ms`;
      const results = queryResult.results_count;
      const notes = queryResult.error ? `Error: ${queryResult.error}`.substring(0, 50) : '-';
      
      lines.push(`| ${query.q} | ${query.mode} | ${status} | ${latency} | ${results} | ${notes} |`);
    });
    lines.push('');
    
    // Recommendations
    lines.push('## Recommendations & Next Steps');
    lines.push('');
    
    reportData.recommendations.forEach((rec, index) => {
      const priority = rec.startsWith('CRITICAL') ? '🔥' : 
                      rec.startsWith('HIGH') ? '⚠️' : 
                      rec.startsWith('MEDIUM') ? '⚡' : 
                      rec.startsWith('NEXT') ? '🔄' : '📋';
      
      lines.push(`${index + 1}. ${priority} ${rec}`);
    });
    
    return lines.join('\n');
  }

  async saveBenchmarkResults(data, reportText) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    // Save raw JSON data
    await fs.writeFile(
      `benchmark-comprehensive-${timestamp}.json`,
      JSON.stringify(data, null, 2)
    );
    
    // Save readable report
    await fs.writeFile(
      `benchmark-comprehensive-report-${timestamp}.md`,
      reportText
    );
    
    console.log(`\n📊 Results saved:`);
    console.log(`   - Raw data: benchmark-comprehensive-${timestamp}.json`);
    console.log(`   - Report: benchmark-comprehensive-report-${timestamp}.md`);
    
    return {
      json: `benchmark-comprehensive-${timestamp}.json`,
      markdown: `benchmark-comprehensive-report-${timestamp}.md`
    };
  }
}

// Run the benchmark if this script is executed directly
if (require.main === module) {
  const benchmark = new BenchmarkReportGenerator();
  
  benchmark.generateComprehensiveReport()
    .then(async (results) => {
      const report = benchmark.generateMarkdownReport(results);
      console.log('\n' + '='.repeat(50));
      console.log('COMPREHENSIVE BENCHMARK COMPLETE');
      console.log('='.repeat(50));
      console.log(report);
      
      const savedFiles = await benchmark.saveBenchmarkResults(results, report);
      
      console.log('\n🎯 NEXT STEPS SUMMARY:');
      console.log('Based on this benchmark analysis, your immediate priorities should be:');
      console.log('');
      
      results.recommendations.filter(r => r.includes('CRITICAL') || r.includes('HIGH')).forEach((rec, i) => {
        console.log(`${i + 1}. ${rec}`);
      });
      
      if (results.recommendations.filter(r => r.includes('CRITICAL') || r.includes('HIGH')).length === 0) {
        console.log('✅ No critical issues found - system ready for continued development');
      }
      
      return savedFiles;
    })
    .catch(error => {
      console.error('Benchmark failed:', error);
      process.exit(1);
    });
}

module.exports = BenchmarkReportGenerator;